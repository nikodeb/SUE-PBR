from .base import AbstractNegativeSampler
from tqdm import trange
from collections import Counter
import numpy as np

class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)

        item_list, item_prob_list = self.item_probabilities_by_popularity()
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            temp_smpl = np.random.choice(item_list, self.sample_size*2, replace=False, p=item_prob_list)
            samples = [t for t in temp_smpl if t not in seen]
            while len(samples) < self.sample_size:
                temp_smpl = np.random.choice(item_list, self.sample_size*2, replace=False, p=item_prob_list)
                samples.extend([t for t in temp_smpl if t not in seen and t not in samples])
            samples = samples[:self.sample_size]
            negative_samples[user] = samples
        return negative_samples

    # def generate_negative_samples(self):
    #     popular_items = self.items_by_popularity()
    #
    #     negative_samples = {}
    #     print('Sampling negative items')
    #     for user in trange(self.user_count):
    #         seen = set(self.train[user])
    #         seen.update(self.val[user])
    #         seen.update(self.test[user])
    #
    #         samples = []
    #         for item in popular_items:
    #             if len(samples) == self.sample_size:
    #                 break
    #             if item in seen:
    #                 continue
    #             samples.append(item)
    #
    #         negative_samples[user] = samples
    #
    #     return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items

    def item_probabilities_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        total = sum(popularity.values())

        items = []
        probs = []
        for key in popularity:
            items.append(key)
            probs.append(popularity[key] / total)
        return items, probs
