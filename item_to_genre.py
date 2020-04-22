import pandas as pd


def item_to_genre(item, data_size='1m'):
    if data_size == '1m' or data_size == '100k':
        data_size = 'ml-' + data_size
        file_dir = 'movielens/' + data_size + '.movies'
    elif data_size == 'ymovie':
        file_dir = 'ymovies/clean_genre.txt'
    movies = pd.read_csv(file_dir, header=0, index_col=0)
    genre = movies.loc[item]
    return genre

def get_genre(data_size):
    data_size = 'ml-' + data_size
    file_dir = 'movielens/' + data_size + '.movies'
    movies = pd.read_csv(file_dir, header=0)
    items = movies.iloc[:, 0].values
    genres = movies.iloc[:, 1:].values
    return (items, genres)


# get_genre('ml-100k')
