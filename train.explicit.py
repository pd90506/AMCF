from amcf import AMCF
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from dataset import get_data
import numpy as np
from item_to_genre import item_to_genre
from evaluate import XEval
from scipy.stats import ttest_1samp


class Args(object):
    """Used to generate different sets of arguments"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = '100k' # '100k', '1m'， 'ymovie'
        self.epochs = 100
        self.batch_size = 256
        self.num_asp = 18 # ml:18
        self.e_dim = 80
        # self.mlp_dim = [64, 32, 16]
        self.reg = 1e-1
        self.bias_reg = 3e-3
        self.asp_reg = 3e-3
        # self.num_neg = 4
        self.lr = 7e-3
        self.bias_lr = 7e-3
        self.asp_lr = 7e-2
        self.lambda1 = 3e-2 # 5e-2
        # self.loss_weights = [1, 1, 1]


def train(model, trainloader, testloader, evaluator, optimizer, criterion, device, args):
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_mse_loss = 0.0
        running_sim_loss = 0.0
        epoch_size = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # get genre information from item id
            item_asp = item_to_genre(inputs[:, 1], data_size=args.dataset).values
            item_asp = torch.Tensor(item_asp).to(device)
            # inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device)
            user_inputs = inputs[:, 0]
            item_inputs = inputs[:, 1]
            epoch_size += len(user_inputs) # calculate the total number of samples in an epoch

            optimizer.zero_grad()
            outputs, cos_sim, pref = model(user_inputs, item_inputs, item_asp) # cos_sim stands for a distance measure.
            outputs = outputs.flatten()
            cos_sim = cos_sim.flatten()


            mse_loss = criterion(outputs, labels.to(torch.float)) # to float
            sim_loss = cos_sim.sum()
            # combined loss
            loss = mse_loss + (args.lambda1 * sim_loss)  
            loss.backward()
            optimizer.step()
            # collect running losses
            running_loss += (mse_loss + (args.lambda1 * sim_loss)).data
            running_mse_loss += mse_loss.data
            running_sim_loss += sim_loss.data
        # total loss
        epoch_loss = running_loss / epoch_size
        # rmse loss
        epoch_mse_loss = running_mse_loss / epoch_size
        rmse = np.sqrt(epoch_mse_loss.cpu().numpy())
        # sim loss
        epoch_sim_loss = running_sim_loss / epoch_size
        print("Epoch {:d}: the training RMSE loss: {:.4f}, item embedding similarity: {:.4f}".format(epoch, rmse, epoch_sim_loss))
        print("Total loss: {:.4f}".format(epoch_loss))
        val_rmse = test(model, testloader, evaluator, criterion, device, args)
        model.train()
    print(30*'+' + 'training completed!' + 30*'+')
    
    return model


def test(model, testloader, evaluator, criterion, device, args):
    model.eval()
    for epoch in range(1):
        running_loss = 0.0
        running_l1loss = 0.0
        running_cos_sim = 0.0
        epoch_size = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            # get genre information from item id
            item_asp = item_to_genre(inputs[:, 1], data_size=args.dataset).values
            item_asp = torch.Tensor(item_asp).to(device)

            # get user list tensor in CPU
            user_in_batch = inputs[:, 0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            user_inputs = inputs[:, 0]
            item_inputs = inputs[:, 1]
            epoch_size += len(user_inputs) # calculate the total number of samples in an epoch

            # optimizer.zero_grad()
            outputs, cos_sim, pref = model(user_inputs, item_inputs, item_asp)
            outputs = outputs.flatten()
            # loss = get_sse(outputs.data.cpu().numpy(), labels.data.cpu().numpy())
            loss = criterion(outputs, labels.to(torch.float)) # to float
            l1loss = nn.L1Loss(reduction='sum')
            total_l1 = l1loss(outputs, labels.to(torch.float))

            running_loss += loss.data
            running_l1loss += total_l1


        epoch_loss = running_loss / epoch_size # get the average loss
        mae = running_l1loss / epoch_size
        rmse = np.sqrt(epoch_loss.cpu().numpy()) # get RMSE by sqrt


        print("The validation RMSE loss: {:.4f}".format(rmse))
        print("The validation MAE loss: {:.4f}".format(mae))

        if args.dataset == '100k':
            users = torch.tensor(range(943), dtype=torch.long).to(device)
            u_pred = model.predict_pref(users)
        elif args.dataset == '1m':
            users = torch.tensor(range(6040), dtype=torch.long).to(device)
            u_pred = model.predict_pref(users)
        elif args.dataset == 'ymovie':
            users = torch.tensor(range(7642), dtype=torch.long).to(device)
            u_pred = model.predict_pref(users)
        else:
            raise Exception("cannot handle right now!!!")

        K = 5
        M = 3
        top_K_acc, bottom_K_acc = evaluator.get_top_K_pos(users, u_pred, K, M)
        print("top {:d} at {:d} aspect accuracy: {:.4f}, \n bottom: {:.4f}".format(M, K, top_K_acc, bottom_K_acc))

        K = 3
        M = 1
        top_K_acc, bottom_K_acc = evaluator.get_top_K_pos(users, u_pred, K, M)
        print("top {:d} at {:d} aspect accuracy: {:.4f}, \n bottom: {:.4f}".format(M, K, top_K_acc, bottom_K_acc))


        cos_sim = evaluator.get_cos_sim(users, u_pred)#.mean()
        p_value = ttest_1samp(cos_sim.cpu().data, 0)
        print("average cos_sim is: {:.4f}".format(cos_sim.mean()))
        print("the p value: {:f}".format(p_value[1]))

    
    return rmse, mae



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = Args()
    # determine data size
    if args.dataset == '1m':
        num_users = 6040
        num_items = 3952  
    elif args.dataset == '100k':
        num_users = 943
        num_items = 1682
    elif args.dataset == 'ymovie':
        num_users = 7643
        num_items = 11916

    # data_loaders contains all K-Fold train_loaders and test_loaders
    data_loaders = get_data(size=args.dataset, batch_size=args.batch_size)
    K = len(data_loaders)
    running_rmse = 0.0
    running_mae = 0.0
    fold = 1
    evaluator = XEval(dataset=args.dataset)
    # load datasets to perform K-Fold cross validation
    for trainloader, testloader in data_loaders:
        print("Fold {:d} / {:d}".format(fold, K))
        # Build model
        model = AMCF(num_user=num_users, num_item=num_items, num_asp=args.num_asp, e_dim=args.e_dim)
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        
        # set parameter learning rates and regularizations
        params_dict = dict(model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key == 'i_bias' or key == 'u_bias':
                params += [{'params':[value],'lr':args.bias_lr, 'weight_decay':args.bias_reg}]
            elif key =='asp_emb.W':
                params += [{'params':[value],'lr':args.asp_lr, 'weight_decay':args.asp_reg}]
            else:
                params += [{'params':[value],'lr':args.lr, 'weight_decay':args.reg}]
        optimizer = SGD(params, lr=args.lr, weight_decay=args.reg)
        
        criterion = nn.MSELoss(reduction='sum')

        # calculate validation losses
        fitted_model = train(model, trainloader, testloader, evaluator, optimizer, criterion, device, args)
        val_rmse, val_mae = test(fitted_model, testloader, evaluator, criterion, device, args)
        running_rmse += val_rmse
        running_mae += val_mae
        fold += 1
    
    print('The overall {:d}-fold cross validation RMSE: {:.4f}'.format(K, running_rmse/K))
    print('The overall {:d}-fold cross validation MAE: {:.4f}'.format(K, running_mae/K))

if __name__ == '__main__':
    main()