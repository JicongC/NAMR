import os
import sys
import json
import time
import heapq
import numpy as np
import multiprocessing
import tensorflow as tf
from LD import Data
from AFA import AdaptiveAttentionLayer
from NA import VisualNoiseAwareModule, TextualNoiseAwareModule


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataset = 'baby'
data_path = '../Data/'
save_path = f'./{dataset}_best_result.json'
batch_size = 2048
embed_size = 128
epoch = 500
data_generator = Data(dataset=dataset, data_path=data_path, batch_size=batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
Ks = np.arange(1, 21)
n_layers = 3
temp = 0.04
decay = 0.001
interval = 10
lambda_v = 0.4
lr = 0.001


class Model(object):
    def __init__(self, data_config, img_feat, text_feat, d1, d2):
        self.d1 = d1
        self.d2 = d2
        self.n_fold = 10
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.norm_adj = data_config['norm_adj']
        self.norm_adj_m = data_config['norm_adj_m']
        self.emb_dim = data_config['embed_size']
        self.batch_size = data_config['batch_size']
        self.n_layers = data_config['n_layers']
        self.decay = data_config['decay']
        self.lr = data_config['lr']
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.weights = self._init_weights()
        self.um_v = self.weights['user_embedding_v']
        self.um_t = self.weights['user_embedding_t']
        self.user_embedding = self.weights['user_embedding']
        self.item_embedding = self.weights['item_embedding']
        self.raw_v = tf.matmul(img_feat, self.weights['w1_v'])
        self.raw_t = tf.matmul(text_feat, self.weights['w1_t'])
        # ===================== 特征净化模块 =====================
        self.noise_aware_v = VisualNoiseAwareModule(self.emb_dim, "visual")
        self.noise_aware_t = TextualNoiseAwareModule(self.emb_dim, "textual")
        self.visual_att = AdaptiveAttentionLayer(units=self.emb_dim, name="visual_att")
        self.textual_att = AdaptiveAttentionLayer(units=self.emb_dim, name="textual_att")
        self.fusion_att = AdaptiveAttentionLayer(units=self.emb_dim, name="fusion_att")
        self.dw_v = self.noise_aware_v(self.raw_v)
        self.dw_t = self.noise_aware_t(self.raw_t)
        raw_v = self.raw_v * self.dw_v
        raw_t = self.raw_t * self.dw_t
        raw_v_3d = tf.expand_dims(raw_v, axis=1)
        raw_t_3d = tf.expand_dims(raw_t, axis=1)
        att_v_3d = self.visual_att(raw_v_3d)
        att_t_3d = self.textual_att(raw_t_3d)
        fused_v_3d, fused_t_3d = self.fusion_att([att_v_3d, att_t_3d])
        self.im_v = tf.squeeze(fused_v_3d, axis=1)
        self.im_t = tf.squeeze(fused_t_3d, axis=1)
        # ==================== 交互维度嵌入 ====================
        self.ua_embeddings, self.ia_embeddings = self._create_norm_embed()
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.user_embedding, self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.item_embedding, self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.item_embedding, self.neg_items)
        # =================== 多模态维度嵌入 ===================
        self.ua_embeddings_v, self.ia_embeddings_v = self._create_norm_embed_v()
        self.u_g_embeddings_v = tf.nn.embedding_lookup(self.ua_embeddings_v, self.users)
        self.pos_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.pos_items)
        self.neg_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.neg_items)
        self.u_g_embeddings_v_pre = tf.nn.embedding_lookup(self.um_v, self.users)
        self.pos_i_g_embeddings_v_pre = tf.nn.embedding_lookup(self.im_v, self.pos_items)
        self.neg_i_g_embeddings_v_pre = tf.nn.embedding_lookup(self.im_v, self.neg_items)
        self.ua_embeddings_t, self.ia_embeddings_t = self._create_norm_embed_t()
        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.pos_items)
        self.neg_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.neg_items)
        self.u_g_embeddings_t_pre = tf.nn.embedding_lookup(self.um_t, self.users)
        self.pos_i_g_embeddings_t_pre = tf.nn.embedding_lookup(self.im_t, self.pos_items)
        self.neg_i_g_embeddings_t_pre = tf.nn.embedding_lookup(self.im_t, self.neg_items)
        self.user_embed = tf.nn.l2_normalize(self.u_g_embeddings, axis=1)
        self.user_embed_v = tf.nn.l2_normalize(self.u_g_embeddings_v, axis=1)
        self.user_embed_t = tf.nn.l2_normalize(self.u_g_embeddings_t, axis=1)
        self.item_embed = tf.nn.l2_normalize(self.pos_i_g_embeddings, axis=1)
        self.item_embed_v = tf.nn.l2_normalize(self.pos_i_g_embeddings_v, axis=1)
        self.item_embed_t = tf.nn.l2_normalize(self.pos_i_g_embeddings_t, axis=1)
        # ===================== 评分预测 =====================
        self.batch_ratings = (
                tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True) +
                lambda_v * tf.matmul(self.u_g_embeddings_v, self.pos_i_g_embeddings_v, transpose_a=False, transpose_b=True) +
                tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t, transpose_a=False, transpose_b=True)
        )
        # =================== 添加额外损失项 ==================
        entropy_loss_v = -tf.reduce_mean(self.dw_v * tf.log(self.dw_v + 1e-8))
        entropy_loss_t = -tf.reduce_mean(self.dw_t * tf.log(self.dw_t + 1e-8))
        reward_loss = tf.reduce_mean(tf.square(tf.maximum(self.dw_v - 0.7, 0))) + tf.reduce_mean(tf.square(tf.maximum(self.dw_t - 0.7, 0)))
        inter_modal_alignment = tf.reduce_mean(tf.square(self.dw_v - self.dw_t))
        self.extra_loss = 0.01 * (entropy_loss_v + entropy_loss_t + reward_loss + inter_modal_alignment)
        # ===================== 损失函数 =====================
        self.mf_loss, self.mf_loss_m, self.emb_loss = self.create_bpr_loss()
        self.cl_loss, self.cl_loss_v, self.cl_loss_t = (self.create_cl_loss_cf(), self.create_cl_loss_v(), self.create_cl_loss_t())
        self.total_loss = self.mf_loss + self.mf_loss_m + self.emb_loss + self.extra_loss
        self.total_cl_loss = self.cl_loss + self.cl_loss_v + self.cl_loss_t
        # ==================== 优化器配置 ====================
        mean_dynamic_weight = tf.reduce_mean(self.dw_v + self.dw_t) / 2.0
        adjusted_decay_rate = 0.95 + 0.04 * (1.0 - mean_dynamic_weight)
        self.dynamic_lr = tf.train.exponential_decay(
            learning_rate=self.lr,
            global_step=self.global_step,
            decay_steps=5000,
            decay_rate=adjusted_decay_rate,
            staircase=True
        )
        self.opt_1 = tf.train.AdamOptimizer(learning_rate=self.dynamic_lr).minimize(self.total_loss)
        self.opt_2 = tf.train.AdamOptimizer(learning_rate=self.dynamic_lr / 10).minimize(self.total_cl_loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        all_weights['user_embedding_v'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding_v')
        all_weights['user_embedding_t'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding_t')
        all_weights['w1_v'] = tf.Variable(initializer([self.d1, self.emb_dim]), name='w1_v')
        all_weights['w1_t'] = tf.Variable(initializer([self.d2, self.emb_dim]), name='w1_t')

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))

        return A_fold_hat

    def _create_norm_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = tf.concat([self.user_embedding, self.item_embedding], axis=0)
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings

        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_norm_embed_v(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_m)
        ego_embeddings_v = tf.concat([self.um_v, self.im_v], axis=0)
        for k in range(0, 1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_v))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings_v = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_v, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def _create_norm_embed_t(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_m)
        ego_embeddings_t = tf.concat([self.um_t, self.im_t], axis=0)
        for k in range(0, 1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_t))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings_t = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_t, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def create_bpr_loss(self):
        pos_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.pos_i_g_embeddings_v), axis=1)
        neg_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.neg_i_g_embeddings_v), axis=1)
        pos_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.pos_i_g_embeddings_t), axis=1)
        neg_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.neg_i_g_embeddings_t), axis=1)
        pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)
        regularizer_mf_v = tf.nn.l2_loss(self.u_g_embeddings_v_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_v_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_v_pre)
        regularizer_mf_t = tf.nn.l2_loss(self.u_g_embeddings_t_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_t_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_t_pre)
        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        mf_loss_m = tf.reduce_mean(tf.nn.softplus(-(pos_scores_v - neg_scores_v))) \
                    + tf.reduce_mean(tf.nn.softplus(-(pos_scores_t - neg_scores_t)))
        emb_loss = self.decay * (regularizer_mf + 0.1 * regularizer_mf_t + 0.1 * regularizer_mf_v) / self.batch_size

        return mf_loss, mf_loss_m, emb_loss

    def create_cl_loss_cf(self):
        pos_score_v = tf.reduce_sum(tf.multiply(self.user_embed, self.user_embed_v), axis=1)
        pos_score_t = tf.reduce_sum(tf.multiply(self.user_embed, self.user_embed_t), axis=1)
        neg_score_v_u = tf.matmul(self.user_embed, self.user_embed_v, transpose_a=False, transpose_b=True)
        neg_score_t_u = tf.matmul(self.user_embed, self.user_embed_t, transpose_a=False, transpose_b=True)
        cl_loss_v_u = - tf.reduce_sum(tf.log(tf.exp(pos_score_v / temp) / tf.reduce_sum(tf.exp(neg_score_v_u / temp), axis=1)))
        cl_loss_t_u = - tf.reduce_sum(tf.log(tf.exp(pos_score_t / temp) / tf.reduce_sum(tf.exp(neg_score_t_u / temp), axis=1)))
        pos_score_v = tf.reduce_sum(tf.multiply(self.item_embed, self.item_embed_v), axis=1)
        pos_score_t = tf.reduce_sum(tf.multiply(self.item_embed, self.item_embed_t), axis=1)
        neg_score_v_i = tf.matmul(self.item_embed, self.item_embed_v, transpose_a=False, transpose_b=True)
        neg_score_t_i = tf.matmul(self.item_embed, self.item_embed_t, transpose_a=False, transpose_b=True)
        cl_loss_v_i = - tf.reduce_sum(tf.log(tf.exp(pos_score_v / temp) / tf.reduce_sum(tf.exp(neg_score_v_i / temp), axis=1)))
        cl_loss_t_i = - tf.reduce_sum(tf.log(tf.exp(pos_score_t / temp) / tf.reduce_sum(tf.exp(neg_score_t_i / temp), axis=1)))
        cl_loss = cl_loss_v_u + cl_loss_t_u + cl_loss_v_i + cl_loss_t_i

        return cl_loss

    def create_cl_loss_v(self):
        pos_score_1 = tf.reduce_sum(tf.multiply(self.user_embed_v, self.user_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.user_embed_v, self.user_embed_t), axis=1)
        neg_score_1_u = tf.matmul(self.user_embed_v, self.user_embed, transpose_a=False, transpose_b=True)
        neg_score_2_u = tf.matmul(self.user_embed_v, self.user_embed_t, transpose_a=False, transpose_b=True)
        cl_loss_1_u = - tf.reduce_sum(tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_u / temp), axis=1)))
        cl_loss_2_u = - tf.reduce_sum(tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_u / temp), axis=1)))
        pos_score_1 = tf.reduce_sum(tf.multiply(self.item_embed_v, self.item_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.item_embed_v, self.item_embed_t), axis=1)
        neg_score_1_i = tf.matmul(self.item_embed_v, self.item_embed, transpose_a=False, transpose_b=True)
        neg_score_2_i = tf.matmul(self.item_embed_v, self.item_embed_t, transpose_a=False, transpose_b=True)
        cl_loss_1_i = - tf.reduce_sum(tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_i / temp), axis=1)))
        cl_loss_2_i = - tf.reduce_sum(tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_i / temp), axis=1)))
        cl_loss = cl_loss_1_u + cl_loss_2_u + cl_loss_1_i + cl_loss_2_i

        return cl_loss

    def create_cl_loss_t(self):
        pos_score_1 = tf.reduce_sum(tf.multiply(self.user_embed_t, self.user_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.user_embed_t, self.user_embed_v), axis=1)
        neg_score_1_u = tf.matmul(self.user_embed_t, self.user_embed, transpose_a=False, transpose_b=True)
        neg_score_2_u = tf.matmul(self.user_embed_t, self.user_embed_v, transpose_a=False, transpose_b=True)
        cl_loss_1_u = - tf.reduce_sum(tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_u / temp), axis=1)))
        cl_loss_2_u = - tf.reduce_sum(tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_u / temp), axis=1)))
        pos_score_1 = tf.reduce_sum(tf.multiply(self.item_embed_t, self.item_embed), axis=1)
        pos_score_2 = tf.reduce_sum(tf.multiply(self.item_embed_t, self.item_embed_v), axis=1)
        neg_score_1_i = tf.matmul(self.item_embed_t, self.item_embed_v, transpose_a=False, transpose_b=True)
        neg_score_2_i = tf.matmul(self.item_embed_t, self.item_embed_t, transpose_a=False, transpose_b=True)
        cl_loss_1_i = - tf.reduce_sum(tf.log(tf.exp(pos_score_1 / temp) / tf.reduce_sum(tf.exp(neg_score_1_i / temp), axis=1)))
        cl_loss_2_i = - tf.reduce_sum(tf.log(tf.exp(pos_score_2 / temp) / tf.reduce_sum(tf.exp(neg_score_2_i / temp), axis=1)))
        cl_loss = cl_loss_1_u + cl_loss_2_u + cl_loss_1_i + cl_loss_2_i

        return cl_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.SparseTensor(indices, coo.data, coo.shape)


def test_one_user(x):
    u, rating = x[1], x[0]
    training_items = data_generator.train_items[u]
    user_pos_test = data_generator.test_set[u]
    all_items = set(range(ITEM_NUM))
    candidate_items = list(all_items - set(training_items) - set(user_pos_test))
    test_items = candidate_items + user_pos_test
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = [1 if i in user_pos_test else 0 for i in K_max_item_score]
    total_pos = len(user_pos_test)
    recall, ndcg = [], []
    for K in Ks:
        hits = sum(r[:K])
        recall_val = hits / total_pos if total_pos > 0 else 0.0
        recall.append(recall_val)
        dcg = 0.0
        for pos in range(K):
            if pos < len(r) and r[pos] == 1:
                dcg += 1.0 / np.log2(pos + 2)

        n_rel = min(total_pos, K)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel)) if n_rel > 0 else 0.0
        ndcg_val = dcg / idcg if idcg > 0 else 0.0
        ndcg.append(ndcg_val)

    return {'ndcg': np.array(ndcg), 'recall': np.array(recall)}


def test(sess, model, users, items, batch_size, cores):
    result = {'ndcg': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks))}
    pool = multiprocessing.Pool(cores)
    n_test_users = len(users)
    u_batch_size = batch_size
    n_user_batchs = (n_test_users + u_batch_size - 1) // u_batch_size
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)
        user_batch = users[start:end]
        rate_batch = sess.run(model.batch_ratings, {model.users: user_batch, model.pos_items: items})
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result['ndcg'] += re['ndcg'] / n_test_users
            result['recall'] += re['recall'] / n_test_users

    pool.close()

    return result


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    cores = multiprocessing.cpu_count() // 3
    data_generator.print_statistics()
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['decay'] = decay
    config['lr'] = lr
    config['n_layers'] = n_layers
    config['embed_size'] = embed_size
    config['batch_size'] = batch_size
    adj_norm, adj_norm_m = data_generator.get_adj_mat()
    config['norm_adj'] = adj_norm
    config['norm_adj_m'] = adj_norm_m
    print('shape of adjacency', adj_norm.shape)
    t0 = time.time()
    model = Model(
        data_config=config,
        img_feat=data_generator.imageFeaMatrix,
        text_feat=data_generator.textFeatMatrix,
        d1=data_generator.imageFeaMatrix.shape[1],
        d2=data_generator.textFeatMatrix.shape[1]
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    early_stopping = 0
    best_score = 0
    best_result = {}
    for epoch in range(300):
        t1 = time.time()
        mf_loss, emb_loss, cl_loss = 0., 0., 0.
        n_batch = data_generator.n_train // batch_size + 1
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_u()
            _, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt_1, model.mf_loss, model.emb_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items})
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            _, batch_cl_loss = sess.run(
                [model.opt_2, model.cl_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items,})
            cl_loss += batch_cl_loss

        if np.isnan(mf_loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % interval != 0:
            perf_str = 'Epoch {} [{:.1f}s]: train==[{:.5f} + {:.5f} + {:.5f}]'.format(
                epoch, time.time() - t1, mf_loss, emb_loss, cl_loss / data_generator.n_train)
            print(perf_str)
            continue

        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())
        result = test(sess, model, users_to_test, data_generator.exist_items, batch_size, cores)
        recall = result['recall']
        ndcg = result['ndcg']
        score = recall[19] + ndcg[19]
        if score > best_score:
            best_score = score
            best_result['recall'] = [str(recall[9]), str(recall[19])]
            best_result['ndcg'] = [str(ndcg[9]), str(ndcg[19])]
            best_result['epoch'] = epoch
            with open(save_path, 'w', encoding='utf-8') as f:
                json_str = json.dumps(best_result, indent=4)
                f.write(json_str)

            print('Best Result: Recall@10,20=%.4f,%.4f; NDCG@10,20=%.4f,%.4f' % (recall[9], recall[19], ndcg[9], ndcg[19]))
            early_stopping = 0
        else:
            early_stopping += 1

        t3 = time.time()
        perf_str = 'Epoch %d [%.1fs + %.1fs]: recall@10=%.5f, recall@20=%.5f, ndcg@10=%.5f, ndcg@20=%.5f' % (
            epoch, t2 - t1, t3 - t2, recall[9], recall[19], ndcg[9], ndcg[19])
        print(perf_str)
        if early_stopping == 10:
            break
