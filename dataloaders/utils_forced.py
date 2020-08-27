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