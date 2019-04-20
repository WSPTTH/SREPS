# -*- coding:utf-8 -*-
import random
import os
import json
from tqdm import tqdm
from collections import namedtuple


_RatingSample = namedtuple('_RatingSample', ('user', 'item', 'rating'))
_RecNetSample = namedtuple('_RecNetSample', ('user', 'ip', 'ineg'))
_SocialSample = namedtuple('_SocialSample', ('user', 'up', 'un'))
_Sample = namedtuple('_Sample', ('rating', 'rec_net', 'social'))
_TestSample = namedtuple('_TestSample', ('user', 'item', 'rating'))


class AliasTable(object):

    def __init__(self, neg_pow=0.75):
        """
        Make up the alias table for the negative sample of links
        The distribution is the 0.75 power of the out degree of each node
        :param neg_pow:
        """
        self._neg_pow = neg_pow
        self._is_build = False
        self._probs = None
        self._alias = None

    @staticmethod
    def _alias_table(dist):
        column_num = len(dist)
        alias = [0 for _ in range(column_num)]  # the alias sample
        probs = [pp * column_num for pp in dist]  # the probability of the second
        small, large = [], []  # the column smaller than 1 or large than 1
        for ix, pp in enumerate(probs):
            large.append(ix) if pp > 1.0 else small.append(ix)

        # create mixture from large and small
        while len(small) and len(large):
            # get one small and one large elements, note that lx > 1, sx + lx > 1
            sx = small.pop()
            lx = large.pop()
            alias[sx] = lx  # the alias to pad the small is `lx`
            probs[lx] -= (1 - probs[sx])  # the remain values
            small.append(lx) if probs[lx] < 1.0 else large.append(lx)
        return probs, alias

    def build(self, links):
        """
        build the alias table
        :param links: the (u0, u1) pairs
        """
        # distribution
        net = {}
        for u0, u1 in links:
            if u0 not in net:
                net[u0] = set()
            net[u0].add(u1)
        degree_sum = sum((len(ix) ** self._neg_pow for ix in net.values()))
        distribution = {ux: len(out) / degree_sum for ux, out in net.items()}
        distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        probs, alias = self._alias_table([pp for _, pp in distribution])

        # mapping to the ids
        self._probs, self._alias = [], {}
        for (ux, _), pp, index in zip(distribution, probs, alias):
            self._probs.append((ux, pp))
            self._alias[ux] = distribution[index][0]
        return self

    def sample(self):
        column = random.randrange(len(self._probs))
        uid, prob = self._probs[column]
        res = uid if random.random() < prob else self._alias[uid]
        return res


class Dataset(object):
    """
    the dataset class to control save the data
    """
    def __init__(self, min_rating=None, max_rating=None):
        self._users = None
        self._items = None
        self._ratings = None
        self._links = None
        self._min_rating = min_rating
        self._max_rating = max_rating
        self._alias_table = AliasTable()

    @staticmethod
    def _front_tail(pairs):
        """get the front set and tail set from the pairs"""
        fronts = set((ix[0] for ix in pairs))
        tails = set((ix[1] for ix in pairs))
        return fronts, tails

    @staticmethod
    def _filter_pairs(pairs, fronts, tails):
        """
        filter the paris with fronts and tiles.
        The pair whose front and tail in the corresponding set is consider
        """
        res = [ix for ix in pairs if ix[0] in fronts and ix[1] in tails]
        return res

    def build(self, ratings, links):
        user_r, items = self._front_tail(ratings)
        user_lf, user_lt = self._front_tail(links)
        users = (user_lf | user_lt) & user_r

        self._users = list(users)
        self._items = list(items)
        self._ratings = self._filter_pairs(ratings, self._users, self._items)
        self._links = self._filter_pairs(links, self._users, self._users)
        self._alias_table.build(self._links)
        return self

    def save(self, file):
        total_pairs = {'rating': self._ratings, 'link': self._links,
                       'max': self._max_rating, 'min': self._min_rating}
        # check the path
        base = os.path.split(file)[0]
        if not os.path.exists(base):
            os.makedirs(base)

        with open(file, 'w', encoding='utf-8') as fp:
            json.dump(total_pairs, fp, ensure_ascii=False)
        return self

    def load(self, file):
        with open(file, 'r', encoding='utf-8') as fp:
            total_pairs = json.load(fp)
        ratings = total_pairs['rating']
        links = total_pairs['link']
        self._max_rating = total_pairs['max']
        self._min_rating = total_pairs['min']
        return self.build(ratings, links)

    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

    @property
    def max_rating(self):
        return self._max_rating

    @property
    def min_rating(self):
        return self._min_rating

    def _rating_sample(self, batch_size):
        res = []
        while len(res) < batch_size:
            user, item, rating = self._ratings[random.randrange(len(self._ratings))]
            res.append((user, item, rating))
        return _RatingSample(*list(zip(*res)))

    def _rec_net_sample(self, batch_size, neg_num):
        res = []
        while len(res) < batch_size:
            user, item, _ = self._ratings[random.randrange(len(self._ratings))]
            neg = []
            while len(neg) < neg_num:
                item_n = self._items[random.randrange(len(self._items))]
                if item_n != item:
                    neg.append(item_n)
            res.append((user, item, neg))
        return _RecNetSample(*list(zip(*res)))

    def _social_sample(self, batch_size, neg_num):
        res = []
        while len(res) < batch_size:
            u0, u1 = self._links[random.randrange(len(self._links))]
            neg = []
            while len(neg) < neg_num:
                un = self._alias_table.sample()
                if un != u0 and un != u1:
                    neg.append(un)
            res.append((u0, u1, neg))
        return _SocialSample(*list(zip(*res)))

    def batch(self, batch_size, neg_num):
        """generate the batch data"""
        while True:
            rating = self._rating_sample(batch_size)
            rec_net = self._rec_net_sample(batch_size // (neg_num + 1) + 1, neg_num)
            social = self._social_sample(batch_size // (neg_num + 1) + 1, neg_num)
            yield _Sample(rating, rec_net, social)

    def test_sample(self, batch_size):
        res = []
        for u, i, r in self._ratings:
            res.append((u, i, r))
            if len(res) == batch_size:
                yield _TestSample(*list(zip(*res)))
                res = []
        if len(res):
            yield _TestSample(*list(zip(*res)))


class DatasetSpliter(object):
    """
    split the hole data set into train and dev
    """
    def __init__(self, ratio=0.2):
        """
        :param ratio: the ratio of the dev dataset
        """
        self._ratio = ratio

    @staticmethod
    def _read_rating(rating_file):
        res = []
        with open(rating_file, 'r', encoding='utf-8') as fp:
            for line in tqdm(fp, desc='reading ratings from `{}`'.format(rating_file), ascii=True):
                user, item, rating = line.strip().split(' ')
                res.append((user, item, float(rating)))
        return res

    @staticmethod
    def _read_links(link_file):
        res = []
        with open(link_file, 'r', encoding='utf-8') as fp:
            for line in tqdm(fp, desc='reading links from `{}`'.format(link_file), ascii=True):
                res.append(line.strip().split())
        return res

    @staticmethod
    def _filter_dev(dev_ratings, train_rating):
        """filter the cold-start users and items (which is not considered in this paper)"""
        user_set = set((ix[0] for ix in train_rating))
        item_set = set((ix[1] for ix in train_rating))
        res = [ix for ix in dev_ratings if ix[0] in user_set and ix[1] in item_set]
        return res

    def split(self, rating_file, link_file):
        # read files
        ratings = self._read_rating(rating_file)
        links = self._read_links(link_file)

        # get the max and min ratings
        max_rating = max((ix[-1] for ix in ratings))
        min_rating = min((ix[-1] for ix in ratings))

        # split ratings
        random.shuffle(ratings)
        train_num = int((1 - self._ratio) * len(ratings))
        train_ratings = ratings[:train_num]
        dev_ratings = self._filter_dev(ratings[train_num:], train_ratings)

        train_dataset = Dataset(min_rating=min_rating, max_rating=max_rating).build(train_ratings, links)
        dev_dataset = Dataset(min_rating=min_rating, max_rating=max_rating).build(dev_ratings, links)
        return train_dataset, dev_dataset
