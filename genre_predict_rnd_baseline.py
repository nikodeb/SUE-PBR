import pickle
import os
import numpy as np

def generate_genre_distribution(train):
    genres_sum = train.sum(axis=0)
    distribution = genres_sum/np.sum(genres_sum)
    return distribution


def convert_dict_dataset_to_numpy(dataset, genres_len):
    dataset_list = []
    for user, cnts_dict in dataset.items():
        user_list = [0] * genres_len
        for genre, cnt in cnts_dict.items():
            user_list[genre] += cnt
        dataset_list.append(user_list)
    return np.array(dataset_list)


def get_most_watched_genre(dataset):
    most_viewed = np.argmax(np.sum(dataset, axis=0))
    return most_viewed


def calculate_accuracy_rnd_smpl(dataset, genres_distribution):
    labels = np.argmax(dataset, axis=1)
    logit_samples = np.random.multinomial(1, genres_distribution, size=dataset.shape[0])
    predictions = np.argmax(logit_samples, axis=1)
    accuracy = (predictions == labels).sum() / dataset.shape[0]
    return accuracy


def calculate_accuracy_most_watched(dataset, most_watched, genres_len):
    labels = np.argmax(dataset, axis=1)
    logit_samples = np.zeros((dataset.shape[0], genres_len))
    logit_samples[:,most_watched] = 1
    predictions = np.argmax(logit_samples, axis=1)
    accuracy = (predictions == labels).sum() / dataset.shape[0]
    return accuracy


dataset_path = 'Data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc0-splitleave_one_out/dataset.pkl'
dataset = pickle.load(open( dataset_path, "rb" ))

gmap = dataset['gmap']
genres_len = len(gmap)

train = convert_dict_dataset_to_numpy(dataset['u2g_train'], genres_len)
val = convert_dict_dataset_to_numpy(dataset['u2g_val'], genres_len)
test = convert_dict_dataset_to_numpy(dataset['u2g_test'], genres_len)

genre_distribution = generate_genre_distribution(train)

total_train = 0
total_val = 0
total_test = 0
repeat = 50

for r in range(repeat):
    train_acc = calculate_accuracy_rnd_smpl(train, genre_distribution)
    val_acc = calculate_accuracy_rnd_smpl(val, genre_distribution)
    test_acc = calculate_accuracy_rnd_smpl(test, genre_distribution)

    total_train += train_acc
    total_val += val_acc
    total_test += test_acc

print('\nResults: Random sampling by popularity')
print('Train Accuracy: %f \t Val Accuracy: %f \t Test Accuracy: %f' % (total_train/repeat, total_val/repeat, total_test/repeat))

# ======================================================================================================================

most_viewed_genre = get_most_watched_genre(train)
train_acc = calculate_accuracy_most_watched(train, most_viewed_genre, genres_len)
val_acc = calculate_accuracy_most_watched(val, most_viewed_genre, genres_len)
test_acc = calculate_accuracy_most_watched(test, most_viewed_genre, genres_len)
print('\nResults: Always predict most watched')
print('Train Accuracy: %f \t Val Accuracy: %f \t Test Accuracy: %f' % (train_acc, val_acc, test_acc))