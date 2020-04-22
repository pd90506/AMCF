import numpy as np
import torch
import torch.nn.functional as F
from item_to_genre import item_to_genre
import pandas as pd
from sklearn.preprocessing import normalize
from topK import topK, hrK


class XEval(object):
    """
    evaluate the explainability
    """
    def __init__(self, dataset='100k'):
        """
        size: can be either '100k' or '1m', or 'ciao'
        """
        if dataset=='100k' or dataset=='1m':
            path = "movielens/" + 'ml-' + dataset + '.ratings'
            path_i ="movielens/" + 'ml-' + dataset + '.iave'
            path_u ="movielens/" + 'ml-' + dataset + '.uave'
        elif dataset=='ymovie':
            path = 'ymovies/clean_data.txt'
            path_i = "ymovies/ymovie.iave"
            path_u = "ymovies/ymovie.uave"
        else:
            raise Exception('not supported dataset!')

        self.dataset = dataset # the dataset name
        # load rating data
        self.data_df = pd.read_csv(path) # dataframe
        self.data = self.data_df.values[:, 0:3] # numpy, note here still float type
        # load averages
        self.i_ave_df = pd.read_csv(path_i, index_col=0) # item id as index
        self.u_ave_df = pd.read_csv(path_u, index_col=0)
        
        ave_dict = {'100k': 3.530, '1m': 3.620, 'ymovie': 4.1}
        self.ave = ave_dict[self.dataset] # the global average
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_general_pref(self, uid):
        """
        given uids, output the preference vectors
        uid is a tensor [batch_size, 1]
        """
        # u_rated_item = self.data[np.isin(self.data[:, 0], uid)]
        # u_rated_asp = item_to_genre(u_rated_item[:, 1], data_size=self.dataset)
        # u_concat = np.concatenate((u_rated_item, u_rated_asp), axis=1)
        # u_pref = np.sum(u_concat, axis=0, where=(u_concat[:,0]==0))
        asp_list = []
        uid = uid.cpu().numpy()
        for u in uid:
            u_rated_item = self.data[self.data[:, 0]==u]
            # aspects for movies rated by a user
            u_rated_asp = item_to_genre(u_rated_item[:, 1], data_size=self.dataset)
            u_rated_asp = np.nan_to_num(u_rated_asp) # important, avoid nan
            u_rated_asp = u_rated_asp.astype(float) # for later calculation
            # ratings given by a user
            u_rating = u_rated_item[:, 2].astype(float)

            ave_rating = u_rating.mean()
            weights = (u_rating - ave_rating) / 5.0 # minus global average, 5 scale
            # weights = weights / weights.sum()
            # weighted sum over movies
            u_pref = np.dot(u_rated_asp.T, weights).T
            # u_pref_norm = np.linalg.norm(u_pref, ord=2)
            # u_pref = u_pref / u_pref_norm
            u_pref = normalize(u_pref[:,np.newaxis], axis=0).ravel()
            u_pref = u_pref.reshape([1, -1])
            asp_list.append(u_pref)
        pref = np.concatenate(asp_list, axis=0)

        return pref


    def get_u_ave(self):
        user = self.data[:, 0].astype(int)
        user = np.unique(user)
        u_ave_list = []
        for u in user:
            u_rated_item = self.data[self.data[:, 0]==u]
            u_rating = u_rated_item[:, 2].astype(float)
            u_ave_rating = u_rating.mean()
            u_ave_list.append(u_ave_rating)
        u_ave = np.array(u_ave_list)
        # name = ['uid', 'ave']
        data = {'uid': user, 'ave': u_ave}
        df = pd.DataFrame(data)
        if self.dataset == '100k':
            path = 'movielens/ml-100k.uave'
        elif self.dataset == '1m':
            path = 'movielens/ml-1m.uave'
        elif self.dataset == 'ymovie':
            path = 'ymovies/ymovie.uave'
        df.to_csv(path, index=False, float_format='%.3f')


    def get_i_ave(self):
        item = self.data[:, 1].astype(int)
        item = np.unique(item)
        i_ave_list = []
        for i in item:
            i_rated_user = self.data[self.data[:, 1]==i]
            i_rating = i_rated_user[:, 2].astype(float)
            i_ave_rating = i_rating.mean()
            i_ave_list.append(i_ave_rating)
        i_ave = np.array(i_ave_list)
        # name = ['uid', 'ave']
        data = {'mid': item, 'ave': i_ave}
        df = pd.DataFrame(data)
        if self.dataset == '100k':
            path = 'movielens/ml-100k.iave'
        elif self.dataset == '1m':
            path = 'movielens/ml-1m.iave'
        elif self.dataset == 'ymovie':
            path = 'ymovies/ymovie.iave'
        df.to_csv(path, index=False, float_format='%.3f')
    

    def get_all_ave(self):
        # this function is for average calculation purpose
        ratings = self.data[:, 2].astype(float)
        ave = ratings.sum()
        print(ave)
        # '100k'： 3.530
        

    def get_u_pref(self, uid):
        ave = self.ave # global average
        df = self.data_df

        u_rated = df.loc[df['uid'].isin(uid)]
        item_rated = u_rated['mid']
        item_ave = self.i_ave_df.loc[item_rated.values].values
        item_bias = item_ave - self.ave
        user_ave = self.u_ave_df.loc[u_rated['uid'].values].values
        user_bias = user_ave - self.ave
        weight = u_rated[['rating']].values - (self.ave + item_bias + user_bias)
        weight = weight.flatten()
        # u_rated['weight'] = weight
        u_rated_asp = item_to_genre(item_rated, data_size=self.dataset).values
        # calculate the weighted rating
        u_pref = np.multiply(u_rated_asp.T, weight).T / 5.0
        u_pref_list = u_pref.tolist()
        u_rated['asp'] = u_pref_list
        u_rated['asp'] = u_rated['asp'].apply(lambda x: np.array(x)) # convert to array
        # u_rated['asp'] = u_rated['asp'].multiply(weight)
        u_rated = u_rated[['uid', 'asp']]
        pref = u_rated.groupby(['uid']).sum()
        # if self.dataset == '100k':
        #     path = 'movielens/ml-100k.upref'
        # pref.to_csv(path, index=False)
        pref_list = pref['asp'].tolist()
        pref_ary = np.array(pref_list)
        return pref_ary


        
    def get_cos_sim(self, uid, predicted):
        """
        predicted: a torch tensor [batch, num_asp]
        uid: a torch tensor [batch]
        """
        pref = self.get_u_pref(uid)
        # convert to tensor.cuda
        pref = torch.tensor(pref, dtype=torch.float).to(self.device)
        pref = F.normalize(pref, p=1, dim=-1)
        sim = F.cosine_similarity(pref, predicted, dim=-1)
        return sim


    def get_specific_cos_sim(self, uid, asp, predicted):
        """
        predicted: a torch tensor [batch, num_asp]
        uid: a torch tensor [batch]
        """
        pref = self.get_u_pref(uid)
        # convert to tensor.cuda
        pref = torch.tensor(pref, dtype=torch.float).to(self.device)
        pref = F.normalize(pref, p=1, dim=-1)
        spec_pref = pref * asp
        spec_pred = predicted * asp
        sim = F.cosine_similarity(spec_pred, spec_pref, dim=-1)
        return sim


    def get_rank(self, pref): # pref is a (batch, num_asp) tensor
        pref = np.abs(pref)
        rank = np.argsort(pref, axis=-1)
        return rank

    def get_top_K_pos_(self, pref, pred, K=3): # top K aspects thats are positive
        """
        pref: given preference
        pred: predicted preference
        """
        pref_rank = self.get_rank(pref)
        pred_rank = self.get_rank(pred)
        pref_top_K = (pref_rank < K).astype(float)
        pred_top_K = (pred_rank < K).astype(float)
        acc_top_K = np.multiply(pref_top_K, pred_top_K).sum(axis=-1) / float(K)
        return acc_top_K

    def get_top_K_pos(self, uid, pred, K=5, M=3):
        pref = self.get_u_pref(uid)
        pred = pred.cpu().data.numpy()
        # return self.get_top_K_pos_(pref, pred, K)
        if self.dataset == '100k':
            num_user = 643
        elif self.dataset == '1m':
            num_user = 6040
        elif self.dataset == 'ymovie':
            num_user = 7642
        return topK(pref, pred, K, M, num_user)

    def get_hr_K(self, uid, pred, K=3):
        pref = self.get_u_pref(uid)
        pred = pred.cpu().data.numpy()
        return hrK(pref, pred, K)

        

if __name__ == '__main__':
    xeval = XEval(dataset='1m')
    # xeval.get_general_pref(torch.tensor([0, 1]))
    a = xeval.get_u_ave()
    b = xeval.get_i_ave()
    # c = xeval.get_top_K_pos(a, b)