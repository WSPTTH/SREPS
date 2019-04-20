import json
import os
import numpy as np
from tqdm import tqdm
from evaluation import evaluate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SREPS(object):

    def __init__(self, dataset, dim=5, ldim=5, udim=5, neg_num=5, alpha=0.2, beta=0.3,
                 lr=0.01, batch=32, lam=0.05, iter_num=6e4, save_path='', save_step=2e3):
        """
        :param dataset: the dataset class
        :param dim: the dimension of the rating space
        :param ldim: the dimension of the social space
        :param udim: the dimension of the essential preference space
        :param neg_num: the number of negative samples
        :param alpha: the hyper-parameter of the social network embedding
        :param beta: the hyper-parameter of the recommendation network
        :param lr: the learning rate
        :param batch: the batch size
        :param lam: the regularization parameter
        :param iter_num: the iteration number
        :param save_path: the model save path (and a `.mapping` file will also be saved)
        :param save_step: the save steps
        """
        assert alpha >= 0 and beta >= 0 and alpha + beta <= 1
        self._data = dataset
        self._max_r = dataset.max_rating
        self._min_r = dataset.min_rating

        self._dim = dim
        self._ldim = ldim
        self._udim = udim
        self._neg_num = neg_num
        self._lr = lr
        self._batch = batch
        self._iter_num = int(iter_num)
        self._save_path = save_path
        self._save_step = int(save_step)

        self._lam = lam
        self._alpha = alpha
        self._beta = beta
        self._gamma = 1 - self._alpha - self._beta

        self._initialize()

    def _initialize(self):
        """initialize the parameters"""
        # set up the mapping
        self._U_mapping = {uid: ix for ix, uid in enumerate(self._data.users, 1)}
        self._I_mapping = {iid: ix for ix, iid in enumerate(self._data.items, 1)}

        # the embeddings
        scale = np.sqrt(3)
        self._U = np.random.rand(len(self._U_mapping) + 1, self._udim) * 2 * scale - scale # user
        self._V = np.random.rand(len(self._I_mapping) + 1, self._dim) * 2 * scale - scale  # item (rating)
        self._B = np.random.rand(len(self._I_mapping) + 1, self._dim) * 2 * scale - scale  # item (rec net)

        # transition matrix
        scale_u = np.sqrt(6) / np.sqrt(self._dim + self._udim)
        scale_l = np.sqrt(6) / np.sqrt(self._ldim + self._udim)
        self._MR = np.random.rand(self._dim, self._udim) * scale_u * 2 - scale_u  # rating
        self._MI = np.random.rand(self._dim, self._udim) * scale_u * 2 - scale_u  # rec net
        self._ME = np.random.rand(self._ldim, self._udim) * scale_l * 2 - scale_l  # social first
        self._MC = np.random.rand(self._ldim, self._udim) * scale_l * 2 - scale_l  # social second

    def _rating_step(self, data):
        u_index = np.array([self._U_mapping[ix] for ix in data.user])
        i_index = np.array([self._I_mapping[ix] for ix in data.item])
        ratings = np.array(data.rating).reshape(-1, 1)

        u = self._U[u_index]
        v = self._V[i_index]
        umr = np.dot(u, self._MR.T)  # [batch, dim]
        score = np.sum(umr * v, axis=1, keepdims=True)  # [batch, 1]
        rx = (score - ratings) * self._gamma

        dv = rx * umr + self._lam * v  # the regularization
        du = rx * np.dot(v, self._MR) + self._lam * (u + np.dot(umr, self._MR))
        dmr = np.dot(v.T, rx * u) / self._batch + self._lam * np.dot(umr.T, u)  # due to the mean of batch data
        self._U[u_index] -= self._lr * du
        self._V[i_index] -= self._lr * dv
        self._MR -= self._lr * dmr

    def _rec_net_step(self, data):
        u_index = np.array([self._U_mapping[ix] for ix in data.user])
        p_index = np.array([self._I_mapping[ix] for ix in data.ip]).reshape(-1, 1)
        n_index = np.array([[self._I_mapping[jx] for jx in ix] for ix in data.ineg])  # [batch, neg]
        i_index = np.concatenate((p_index, n_index), axis=-1)  # [batch, neg + 1]

        item_num = i_index.shape[1]
        batch = i_index.shape[0]
        i_index = i_index.reshape(batch * item_num)

        u = self._U[u_index]
        items = self._B[i_index].reshape(batch, item_num, self._dim)  # [batch, neg + 1, dim]
        label = np.zeros((batch, item_num))
        label[:, 0] = 1

        umi = np.dot(u, self._MI.T)  # [batch, dim]
        score = np.sum(np.expand_dims(umi, 1) * items, axis=-1)  # [batch, neg+1]
        score = sigmoid(score)
        rx = - (label - score) * self._beta
        rx = np.expand_dims(rx, 2)

        du = np.sum(rx * np.dot(items, self._MI), axis=1) + self._lam * (u + np.dot(umi, self._MI))
        di = rx * np.expand_dims(umi, 1) + self._lam * items
        sum_item = np.sum(rx * items, axis=1)  # [batch, dim]
        dmi = np.dot(sum_item.T, u) / self._batch + self._lam * np.dot(umi.T, u)

        self._U[u_index] -= self._lr * du
        self._B[i_index] -= self._lr * di.reshape(batch * item_num, self._dim)
        self._MI -= self._lr * dmi

    def _social_step(self, data):
        u_index = np.array([self._U_mapping[ix] for ix in data.user])
        up_index = np.array([self._U_mapping[ix] for ix in data.up]).reshape(-1, 1)
        un_index = np.array([[self._U_mapping[jx] for jx in ix] for ix in data.un])  # [batch, neg]
        s_index = np.concatenate((up_index, un_index), axis=-1)

        s_num = s_index.shape[1]
        batch = s_index.shape[0]
        s_index = s_index.reshape(batch * s_num)

        u = self._U[u_index]
        soc = self._U[s_index].reshape(batch, s_num, self._udim)  # [batch, neg + 1, udim]
        label = np.zeros((batch, s_num))
        label[:, 0] = 1

        ume = np.dot(u, self._ME.T)  # [batch, dim]
        smc = np.dot(soc, self._MC.T)  # [batch, neg + 1, dim]

        score = np.sum(np.expand_dims(ume, 1) * smc, axis=-1)  # [batch, neg+1]
        score = sigmoid(score)
        rx = - (label - score) * self._alpha
        rx = np.expand_dims(rx, 2)

        du = np.sum(rx * np.dot(smc, self._ME), axis=1) + self._lam * (u + np.dot(ume, self._ME))
        ds = rx * np.expand_dims(np.dot(ume, self._MC), 1) + self._lam * (soc + np.dot(smc, self._MC))
        sum_smc = np.sum(rx * smc, axis=1)  # [batch, dim]
        dme = np.dot(sum_smc.T, u) / self._batch + self._lam * np.dot(ume.T, u)
        dmc = np.dot(ume.T, u) + self._lam * np.sum(np.matmul(np.transpose(smc, (0, 2, 1)), soc), axis=0)
        dmc /= self._batch

        self._U[u_index] -= self._lr * du
        self._U[s_index] -= self._lr * ds.reshape(batch * s_num, self._udim)
        self._ME -= self._lr * dme
        self._MC -= self._lr * dmc

    def _train_step(self, data):
        """separately train the three losses"""
        self._rating_step(data.rating)
        self._rec_net_step(data.rec_net)
        self._social_step(data.social)
        # the default value is the mean
        self._U[0] = np.mean(self._U[1:], axis=0)
        self._V[0] = np.mean(self._V[1:], axis=0)
        self._B[0] = np.mean(self._B[1:], axis=0)

    def train(self, evals=None):
        # format evals
        evals = self._check_evalset(evals)
        batch_data = self._data.batch(self._batch, self._neg_num)
        for ix in tqdm(range(self._iter_num), desc='training', ascii=True):
            self._train_step(next(batch_data))
            if (ix + 1) % self._save_step == 0:
                self.save(self._save_path)
                # evaluate
                all_res = self.eval(evals)
                for name, res in all_res:
                    log_base = '[Evaluate at {:6d} Dataset `{}` | '.format(ix + 1, name)
                    str_res = ['{}: {:>2.5f}'.format(kx, vx) for kx, vx in sorted(res.items())]
                    log_values = ' | '.join(str_res)
                    tqdm.write(log_base + log_values)
        self.save(self._save_path)
        return self

    def save(self, path):
        # check path
        base = os.path.split(path)[0]
        if not os.path.exists(base):
            os.makedirs(base)

        # save mapping
        mappings = {'user': self._U_mapping, 'item': self._I_mapping}
        with open(path + '.mapping', 'w', encoding='utf-8') as fp:
            json.dump(mappings, fp)

        # save parameter
        data = {
            'user': self._U, 'item': self._V, 'item_rec': self._B,
            'mr': self._MR, 'mi': self._MI, 'me': self._ME, 'mc': self._MC
        }
        np.savez_compressed(path, **data)

    def load(self, path):
        with open(path + '.mapping', 'r', encoding='utf-8') as fp:
            mappings = json.load(fp)
        self._U_mapping, self._I_mapping = mappings['user'], mappings['item']

        with np.load(path) as data:
            self._U, self._V, self._B = data['user'], data['item'], data['item_rec']
            self._MR, self._MI = data['mr'], data['mi']
            self._ME, self._MC = data['me'], data['mc']
        return self

    def _score(self, user, item):
        u_ind = np.array([self._U_mapping.get(uid, 0) for uid in user])
        i_ind = np.array([self._I_mapping.get(iid, 0) for iid in item])
        u = np.dot(self._U[u_ind], self._MR.T)
        i = self._V[i_ind]
        score = np.sum(u * i, axis=1)
        score = np.clip(score, self._min_r, self._max_r)
        return score

    @staticmethod
    def _check_evalset(evals):
        if evals is not None:
            assert isinstance(evals, (list, dict, tuple))
            if isinstance(evals, (list, tuple)):
                evals = list(enumerate(evals))
            elif isinstance(evals, dict):
                evals = list(evals.items())
        return evals

    def eval(self, evals):
        res = []
        for name, dataset in evals:
            data_gen = dataset.test_sample(self._batch)
            reals, preds = [], []
            for user, item, ratings in data_gen:
                scores = self._score(user, item)
                reals.extend(ratings)
                preds.extend(scores)
            data_res = evaluate(reals, preds)
            res.append((name, data_res))
        return res
