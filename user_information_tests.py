import pickle
import json
import csv
import os
import numpy as np
from operator import itemgetter


def load_preprocessed_dataset(path):
    dataset = pickle.load(open(path, "rb"))
    return dataset


def load_text_files(file_list):
    text_file_list = []
    for file in file_list:
        with open(file) as f:
            text_file_list.append(f.readlines())
    return text_file_list


def generate_inverse_maps(map_list):
    out_map_list = []
    for map in map_list:
        out_map_list.append({v:k for k,v in map.items()})
    return out_map_list


def process_movie_list(movie_list, s_map):
    movie_id_to_name = {}
    movie_id_to_genres = {}
    genres_to_movie_ids = {}
    for movie_line in movie_list:
        if movie_line is None or movie_line.strip() == '':
            continue

        movie_line_split = movie_line.split('::')
        movie_id, movie_name, movie_genres_unclean = movie_line_split[0].strip(), movie_line_split[1].strip(), \
                                                     movie_line_split[2].strip()
        movie_id = int(movie_id)
        if movie_id not in s_map:
            continue

        movie_genres_split = movie_genres_unclean.split('|')
        movie_genres = [g.strip() for g in movie_genres_split]

        movie_id = s_map[movie_id]
        movie_id_to_name[movie_id] = movie_name
        movie_id_to_genres[movie_id] = movie_genres

        for genre in movie_genres:
            if genre not in genres_to_movie_ids:
                genres_to_movie_ids[genre] = []
            genres_to_movie_ids[genre].append(movie_id)

    genres_to_movie_ids = {k: set(v) for k, v in genres_to_movie_ids.items()}

    return movie_id_to_name, movie_id_to_genres, genres_to_movie_ids


def process_user_rating_data(train, movie_id_to_genres, genres_to_movie_ids):
    user_movies_watched = {}
    user_genres_watched = {}
    user_genres_count = {u: {g: 0 for g in genres_to_movie_ids.keys()} for u in train.keys()} #init with full genres set to 0 cnt
    genres_to_users = {g: [] for g in genres_to_movie_ids.keys()}

    for user_id, movies_watched in train.items():
        user_movies_watched[user_id] = set(movies_watched)

        genres = []
        for movie in movies_watched:
            g_list = movie_id_to_genres[movie]
            genres.extend(g_list)
            for g in g_list:
                user_genres_count[user_id][g] += 1
                genres_to_users[g].append(user_id)

        user_genres_watched[user_id] = set(genres)

    genres_to_users = {k: set(v) for k,v in genres_to_users.items()}
    user_movies_watched = {k: set(v) for k, v in user_movies_watched.items()}
    user_genres_watched = {k: set(v) for k, v in user_genres_watched.items()}

    user_genres_percent = {}
    for user in user_genres_count:
        total = 0
        for g, c in user_genres_count[user].items():
            total += c
        user_genres_percent[user] = {g: c/total for g, c in user_genres_count[user].items()}

    return user_movies_watched, user_genres_watched, user_genres_count, genres_to_users, user_genres_percent


def get_filtered_users_genre(user_genres_percent, genre, threshold=0.55):
    users_high_genre = []
    for user, genre_pcts in user_genres_percent.items():
        if genre_pcts[genre] > threshold:
            users_high_genre.append((user, genre_pcts[genre]))

    return sorted(users_high_genre, key=itemgetter(1), reverse=True)

cwd = os.path.abspath(os.getcwd())
dataset_dir = os.path.join(cwd, 'Data', 'ml-1m')
movies_dir = os.path.join(dataset_dir, 'movies.dat')
users_dir = os.path.join(dataset_dir, 'users.dat')
ratings_dir = os.path.join(dataset_dir, 'ratings.dat')
prepro_dataset_dir = os.path.join(cwd, 'Data', 'preprocessed', 'ml-1m_min_rating0-min_uc5-min_sc0-splitleave_one_out', 'dataset.pkl')

preprocessed_dataset = load_preprocessed_dataset(prepro_dataset_dir)
train, val, test, s_map, u_map = preprocessed_dataset['train'], preprocessed_dataset['val'], preprocessed_dataset['test'], preprocessed_dataset['smap'], preprocessed_dataset['umap']
inv_s_map, inv_u_map = generate_inverse_maps([s_map, u_map])

movie_list, users_list, ratings_list = load_text_files([movies_dir, users_dir, ratings_dir])
movie_id_to_name, movie_id_to_genres, genres_to_movie_ids = process_movie_list(movie_list, s_map)

user_movies_watched, user_genres_watched, user_genres_count, genres_to_users, user_genres_percent = \
    process_user_rating_data(train, movie_id_to_genres, genres_to_movie_ids)

users_high_horror = get_filtered_users_genre(user_genres_percent, genre='Horror', threshold=0.5)
users_high_comedy = get_filtered_users_genre(user_genres_percent, genre='Comedy', threshold=0.5)
users_high_documentary = get_filtered_users_genre(user_genres_percent, genre='Documentary', threshold=0.5)

u1_h = users_high_horror[0][0]
u2_d = users_high_documentary[0][0]

u1_h_watched = list(genres_to_movie_ids['Horror'].intersection(user_movies_watched[u1_h]))[:50]
u1_h_not_watched = list(genres_to_movie_ids['Horror'].difference(user_movies_watched[u1_h].union(user_movies_watched[u2_d])))[:100-len(u1_h_watched)]

u2_d_watched = list(genres_to_movie_ids['Documentary'].intersection(user_movies_watched[u2_d]))[:50]
u2_d_not_watched = list(genres_to_movie_ids['Documentary'].difference(user_movies_watched[u1_h].union(user_movies_watched[u2_d])))[:100-len(u2_d_watched)]

print(len(u2_d_watched) + len(u2_d_not_watched))

print(genres_to_movie_ids)
