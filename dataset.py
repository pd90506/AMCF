import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

random_seed = 0

def get_data(size='100k', batch_size=256):
    """
    size: can be either '100k' or '1m', or 'ciao'
    """
    if size=='100k' or size=='1m':
        path = "movielens/" + 'ml-' + size + '.ratings'
    elif size=='ymovie':
        path = 'ymovies/clean_data.txt'
    else:
        raise Exception('not supported dataset!')
    data = pd.read_csv(path)
    data = data.values # convert to numpy array
    inps = data[:, 0:2].astype(int) # get user, item inputs
    tgts = data[:, 2].astype(int) # get rating targets
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    loaders = []
    for train_index, test_index in kf.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # split and convert to tensors
        inps_train = torch.tensor(inps[train_index], dtype=torch.long)
        inps_test = torch.tensor(inps[test_index], dtype=torch.long)
        tgts_train = torch.tensor(tgts[train_index], dtype=torch.long)
        tgts_test = torch.tensor(tgts[test_index], dtype=torch.long)
        # convert to TensorDataset type
        train_set = TensorDataset(inps_train, tgts_train)
        test_set = TensorDataset(inps_test, tgts_test)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        loaders.append([train_loader, test_loader])
    
    # return all loaders for cross validation
    return loaders   



if __name__ == '__main__':
    dataloader = get_data('ciao')
    print()