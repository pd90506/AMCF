import pandas as pd
import requests
import zipfile
from tqdm import tqdm
import math
import os
import random
# from os.path import dirname, abspath
# from sklearn.model_selection import train_test_split


def download_ml(data_size):
    """
    Download movielens dataset of different sizes
    Args:
        data_size: a string that indicate the size of the dataset, e.g 'ml-1m',
        'ml-100k' etc.
    """
    data_urls = {
        'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
    }
    url = data_urls[data_size]
    file_name = 'temp/' + data_size + '.zip'
    # if the data directory already exist, return
    if os.path.isdir(data_size):
        return
    if not os.path.isdir('temp'):
        os.mkdir('temp')

    # if the directory doesn't exist, download and extract data
    with open(file_name, 'wb') as f:
        r = requests.get(url)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length'))
        block_size = 1024
        wrote = 0
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size/block_size),
                         unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
    # unzip to current directory
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall()


def create_csv_data(source_dir='ml-100k', target_dir='movielens'):
    """
    convert original data to csv file and movie to the 'movielens' folder
    args:
        source: source_dir .dat directory
        target: target_dir .csv directory
    """
    if not os.path.isdir(source_dir):
        download_ml(source_dir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    if source_dir == 'ml-100k':
        convert_100k(source_dir, target_dir)
    if source_dir == 'ml-1m':
        convert_1m(source_dir, target_dir)


def convert_100k(source_dir, target_dir):
    """
    convert ml-100k original dataset to csv files
    """
    # set input and output directories
    source_data = source_dir + '/u.data'
    source_movie = source_dir + '/u.item'
    target_data = target_dir + '/ml-100k.ratings'
    target_movie = target_dir + '/ml-100k.movies'

    # from source data to target data
    col_names = ['uid', 'mid', 'rating', 'timestamp']
    data = pd.read_csv(source_data, sep='\t', names=col_names)
    data['uid'] = data['uid'].apply(lambda x: x - 1)
    data['mid'] = data['mid'].apply(lambda x: x - 1)
    data = data.sort_values(by=['uid'], axis=0).reset_index(drop=True)

    data.to_csv(target_data, index=False)

    # from source movie to target movie
    col_names = ['mid', 'title', 'release_date', 'video_release_date',
                 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
    movies = pd.read_csv(source_movie, sep='|', names=col_names,
                         encoding='latin-1')
    movies = movies[['mid', 'Action', 'Adventure',
                     'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']]
    movies['mid'] = movies['mid'].apply(lambda x: x - 1)
    movies.to_csv(target_movie, index=False)


def convert_1m(source_dir, target_dir):
    # set input and output directories
    source_data = source_dir + '/ratings.dat'
    source_movie = source_dir + '/movies.dat'
    target_data = target_dir + '/ml-1m.ratings'
    target_movie = target_dir + '/ml-1m.movies'

    # from source data to target data
    col_names = ['uid', 'mid', 'rating', 'timestamp']
    data = pd.read_csv(source_data, sep='::', names=col_names)
    data['uid'] = data['uid'].apply(lambda x: x - 1)
    data['mid'] = data['mid'].apply(lambda x: x - 1)
    data.to_csv(target_data, index=False)

    # from source movies to target movies
    col_names = ['mid', 'title', 'genre']
    movies = pd.read_csv(source_movie, sep='::', names=col_names,
                         encoding='latin-1')
    # Processing genre_list column
    movies['genre_list'] = movies['genre'].apply(genre_to_int_list)
    new_df = movies['genre'].apply(genre_to_int_list)
    a = new_df.apply(lambda x: pd.Series(x))
    a['mid'] = movies['mid'].apply(lambda x: x - 1)
    col_names = ['Action', 'Adventure',
                 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western', 'mid']
    a.columns = col_names
    
    a = a[['mid', 'Action', 'Adventure',
           'Animation', 'Children', 'Comedy', 'Crime',
           'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
           'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
           'Thriller', 'War', 'Western']]
    a.to_csv(target_movie, index=False)


def genre_to_int_list(genre_string):
    """
    Convert the list of genre names to a list of integer codes
    Args:
        genre_string: a string of genres names.
    """
    GENRES = ('Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western')
    GENRES_LC = tuple((x.lower() for x in GENRES))
    # convert to lower case
    genre_string_lc = genre_string.lower()
    genre_list = []
    for idx in range(len(GENRES_LC)):
        if GENRES_LC[idx] in genre_string_lc:
            genre_list.append(1)
        else:
            genre_list.append(0)
    return genre_list


def sample_negative(ratings):
    """
    return 100 sampled negative items for each user
    args:
        ratings: ratings dataset, a dataframe
    """
    # user_pool = set(ratings['userId'].unique())
    item_pool = set(ratings['mid'].unique())

    interact_status = ratings.groupby('uid')['mid'].apply(set)\
        .reset_index().rename(columns={'mid': 'interacted_items'})
    interact_status['negative_items'] = interact_status['interacted_items']\
        .apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items']\
        .apply(lambda x: random.sample(x, 99))
    return interact_status[['uid', 'negative_samples']]


def split_train_test(ratings):
    """return training set and test set by loo"""
    ratings['rank_latest'] = ratings.groupby(['uid'])['timestamp']\
        .rank(method='first', ascending=False)
    test = ratings[ratings['rank_latest'] == 1]
    train = ratings[ratings['rank_latest'] > 1]
    assert train['uid'].nunique() == test['uid'].nunique()
    return train[['uid', 'mid', 'rating']], test[['uid', 'mid',
                                                  'rating']]


def sample_ml(data_size='ml-100k', target_dir='Data'):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    # define the file names of train, test and negatives
    train_file = target_dir + '/' + data_size + '.train.rating'
    test_file = target_dir + '/' + data_size + '.test.rating'
    test_negatives = target_dir + '/' + data_size + '.test.negative'
    # load the source data file
    source_dir = 'movielens/' + data_size + '.ratings'
    ratings = pd.read_csv(source_dir, header=0)
    negatives = sample_negative(ratings)
    train, test = split_train_test(ratings)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    negatives.to_csv(test_negatives, index=False)


if __name__ == "__main__":
    create_csv_data(source_dir='ml-100k')
    create_csv_data(source_dir='ml-1m')
    sample_ml(data_size='ml-100k')
    sample_ml(data_size='ml-1m')
