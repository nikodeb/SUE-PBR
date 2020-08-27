import pandas

from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    @abstractmethod
    def load_movies_df(self):
        pass

    def load_dataset(self):
        # Download the dataset if required, preprocess the raw dataset files, dump as pickle
        self.preprocess()
        # Use the raw and preprocessed files to generate user genre data
        self.preprocess_user_genres()
        dataset_path = self._get_preprocessed_dataset_path()
        # Load and return the fully processed dataset
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def preprocess_user_genres(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if not dataset_path.is_file():
            print('Main dataset not preprocessed. Please preprocess the main dataset first')
            return

        dataset = pickle.load(dataset_path.open('rb'))
        if 'gmap' in dataset and 'u2g_train' in dataset and 'u2g_val' in dataset and 'u2g_test' in dataset:
            print('User genre dataset already preprocessed. Skipping preprocessing.')
            return

        smap = dataset['smap']
        user_movies_watched = {u: dataset['train'][u] + dataset['val'][u] + dataset['test'][u] for u in
                               dataset['train'].keys()}

        movie_df = self.load_movies_df()
        movie_df = self._explode_movies_df(movie_df)
        movie_df, gmap = self._generate_genre_map(movie_df)
        movie_df = self._clean_movie_list(movie_df, smap)
        user_genre_pct = self._get_user_genre_pct(movie_df, user_movies_watched)
        train, val, test = self.split_user_genres(user_genre_pct)

        dataset['gmap'] = gmap
        dataset['u2g_train'] = train
        dataset['u2g_val'] = val
        dataset['u2g_test'] = test

        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def _explode_movies_df(self, movies_df):
        df = movies_df
        return splitDataFrameList(df, 'genre', '|')

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i + 2 for i, s in enumerate(set(df['sid']))} # +2 for [pad] = 0, [mask] = 1
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting: leave-one-out')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        elif self.args.split == 'leave_two_out':
            print('Splitting: leave-two-out')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-3], items[-3:-1], items[-1:]
            return train, val, test
        else:
            raise NotImplementedError

    def _generate_genre_map(self, df):
        print('Generating genre mapping')
        gmap = {g: i for i, g in enumerate(set(df['genre']))}
        df['genre'] = df['genre'].map(gmap)
        return df, gmap

    def _clean_movie_list(self, df, smap):
        df['sid'] = df['sid'].map(smap)
        df = df.dropna()
        return df

    def _get_user_genre_pct(self, movie_df, user_movies):
        user_df = pandas.DataFrame([[user, movie] for user in user_movies for movie in user_movies[user]])
        user_df.columns = ['uid', 'sid']
        df_inner = pd.merge(movie_df, user_df, on='sid', how='inner')
        pctgs = df_inner.groupby('uid')['genre'].value_counts(normalize=False).reset_index(name='user_genre_pct')

        def convert_to_dict(df):
            dict_y = dict(zip(df['genre'], df['user_genre_pct']))
            return dict_y

        user2genres_df = pctgs.groupby('uid').progress_apply(convert_to_dict).reset_index(0)

        user_genres_pct = {}
        for user in user_movies:
            user_genres_pct[user] = list(user2genres_df.loc[user2genres_df['uid'] == user][0])[0]

        return user_genres_pct

    def split_user_genres(self, u2genres):
        d = [k for k in u2genres.keys()]
        tr = int(0.8 * len(d))
        tst = -int(0.1 * len(d))
        train = {k: u2genres[k] for k in d[:tr]}
        val = {k: u2genres[k] for k in d[tr:tst]}
        test = {k: u2genres[k] for k in d[tst:]}
        return train, val, test

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

    def _get_preprocessed_user_genres_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('user_genres.pkl')