import os
import time
import numpy as np
import random as rd
import scipy.sparse as sp
from collections import defaultdict


def inject_feature_noise(feat_matrix, std, noise_ratio):
    noise_mask = np.random.rand(*feat_matrix.shape) < noise_ratio
    noise = np.random.normal(0.0, std, feat_matrix.shape).astype(np.float32)

    return (feat_matrix + noise_mask * noise).astype(np.float32)



class Data(object):
    def __init__(self, dataset, data_path, batch_size):
        self.path = data_path + dataset
        self.batch_size = batch_size
        self.inter_file = os.path.join(self.path, f'{dataset}.inter')
        self.train_items = {}
        self.test_set = {}
        self.exist_users = []
        self.n_train = 0
        self.n_test = 0
        self.n_users = 0
        self.n_items = 0
        self.process_inter_file(self.inter_file)
        self.exist_items = list(range(self.n_items))
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u in self.train_items:
            for i in self.train_items[u]:
                self.R[u, i] = 1.0

        self.R = self.R.tocsr()
        self.coo_R = self.R.tocoo()
        self.imageFeaMatrix = np.load(os.path.join(self.path, 'image_feat.npy')).astype(np.float32)
        self.textFeatMatrix = np.load(os.path.join(self.path, 'text_feat.npy')).astype(np.float32)
        # self.imageFeaMatrix = inject_feature_noise(
        #     np.load(os.path.join(self.path, 'image_feat.npy')).astype(np.float32),
        #     std=0.05, noise_ratio=0.1
        # )
        # self.textFeatMatrix = inject_feature_noise(
        #     np.load(os.path.join(self.path, 'text_feat.npy')).astype(np.float32),
        #     std=0.05, noise_ratio=0.1
        # )

    def process_inter_file(self, inter_path):
        user_hist = defaultdict(list)
        with open(inter_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue

                u, i, *_ = line.split('\t')
                u, i = int(u), int(i)
                user_hist[u].append(i)
                self.n_items = max(self.n_items, i)
                self.n_users = max(self.n_users, u)

        for u in user_hist:
            items = user_hist[u]
            if len(items) < 2:
                continue

            self.exist_users.append(u)
            train, test = items[:-1], items[-1:]
            self.train_items[u] = train
            self.test_set[u] = test
            self.n_train += len(train)
            self.n_test += len(test)

        self.n_items += 1
        self.n_users += 1

    def get_adj_mat(self):
        origin_file = self.path + '/origin'
        try:
            t1 = time.time()
            if not os.path.exists(origin_file):
                os.mkdir(origin_file)

            norm_adj_mat = sp.load_npz(origin_file + '/adj_mat.npz')
            norm_adj_mat_m = sp.load_npz(origin_file + '/adj_mat_m.npz')
            print('Already loaded adjacency matrices:', norm_adj_mat.shape, norm_adj_mat_m.shape, time.time() - t1)

        except Exception:
            norm_adj_mat, norm_adj_mat_m = self.create_adj_mat()
            sp.save_npz(origin_file + '/adj_mat.npz', norm_adj_mat)
            sp.save_npz(origin_file + '/adj_mat_m.npz', norm_adj_mat_m)

        return norm_adj_mat, norm_adj_mat_m

    def create_adj_mat(self):
        t1 = time.time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[:self.n_users, self.n_users: self.n_users + self.n_items] = R
        adj_mat[self.n_users: self.n_users + self.n_items, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('Created adjacency matrix:', adj_mat.shape, time.time() - t1)
        t2 = time.time()

        def normalized_adj_symetric(adj, d1, d2):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, d1).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            d_inv_sqrt_last = np.power(rowsum, d2).flatten()
            d_inv_sqrt_last[np.isinf(d_inv_sqrt_last)] = 0.
            d_mat_inv_sqrt_last = sp.diags(d_inv_sqrt_last)

            return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt_last).tocoo()

        norm_adj_mat = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), -0.5, -0.3)
        norm_adj_mat_m = normalized_adj_symetric(adj_mat + sp.eye(adj_mat.shape[0]), -0.5, -0.4)
        norm_adj_mat = norm_adj_mat.tocsr()
        norm_adj_mat_m = norm_adj_mat_m.tocsr()
        print('Normalized adjacency matrices:', time.time() - t2)

        return norm_adj_mat.tocsr(), norm_adj_mat_m.tocsr()

    def sample_u(self):
        total_users = self.exist_users
        users = rd.sample(total_users, self.batch_size)

        def sample_pos_items_for_u(u):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]
            return pos_i_id

        def sample_neg_items_for_u(u):
            pos_items = self.train_items[u]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in pos_items:
                    return neg_id

        pos_items, neg_items, pos_users_for_pi, neg_users_for_pi = [], [], [], []
        for u in users:
            pos_i = sample_pos_items_for_u(u)
            neg_i = sample_neg_items_for_u(u)

            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
