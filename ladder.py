# following implementation is based on https://github.com/rinuboney/ladder

import tensorflow as tf
import numpy as np
import math

from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict

import input_data
from common import *

class Model(object):
    def __init__(self, enc_dec_layers, noise_std):
        L = len(enc_dec_layers) - 1  # number of layers
        dim_X = enc_dec_layers[0]
        num_classes = 10

        self.X_L = tf.placeholder(tf.float32, shape=(None, dim_X))
        self.X_U = tf.placeholder(tf.float32, shape=(None, dim_X))
        self.R_L = tf.placeholder(tf.float32, shape=(None,num_classes))

        len_l, len_u = tf.shape(self.X_L)[0], tf.shape(self.X_U)[0]
        join = lambda l, u: tf.concat([l, u], 0)
        #labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
        #unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
        
        labeled = lambda x: x[:len_l, :] if x is not None else x
        unlabeled = lambda x: x[len_l:, :] if x is not None else x
        split_lu = lambda x: (labeled(x), unlabeled(x))
        self.labeled, self.unlabeled, self.split_lu = labeled, unlabeled, split_lu

        X = join(self.X_L, self.X_U)
        #inputs = tf.placeholder(tf.float32, shape=(None, enc_dec_layers[0]))
        #outputs = tf.placeholder(tf.float32, shape=(None, 10))

        def bi(inits, size, name):
            return tf.Variable(inits * tf.ones([size]), name=name)

        def wi(shape, name):
            return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

        shapes = list(zip(enc_dec_layers[:-1], enc_dec_layers[1:]))  # shapes of linear layers

        weights = {'W': [wi(s, "W") for s in shapes],  # Encoder weights
                   'V': [wi(s[::-1], "V") for s in shapes],  # Decoder weights
                   # batch normalization parameter to shift the normalized value
                   'beta': [bi(0.0, enc_dec_layers[l+1], "beta") for l in range(L)],
                   # batch normalization parameter to scale the normalized value
                   'gamma': [bi(1.0, enc_dec_layers[l+1], "beta") for l in range(L)]}

        self.training = tf.placeholder(tf.bool)

        ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        self.bn_assigns = []  # this list stores the updates to be made to average mean and variance


        def batch_normalization(batch, mean=None, var=None):
            if mean is None or var is None:
                mean, var = tf.nn.moments(batch, axes=[0])
            return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

        # average mean and variance of all layers
        running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in enc_dec_layers[1:]]
        running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in enc_dec_layers[1:]]


        def update_batch_normalization(batch, l):
            "batch normalize + update average mean and variance of layer l"
            mean, var = tf.nn.moments(batch, axes=[0])
            assign_mean = running_mean[l-1].assign(mean)
            assign_var = running_var[l-1].assign(var)
            self.bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
            with tf.control_dependencies([assign_mean, assign_var]):
                return (batch - mean) / tf.sqrt(var + 1e-10)


        def encoder(inputs, noise_std):
            h = inputs + tf.random_normal(tf.shape(inputs)) * (0 if noise_std is None else noise_std) # add noise to input
            d = {}  # to store the pre-activation, activation, mean and variance for each layer
            # The data for labeled and unlabeled examples are stored separately
            d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
            d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
            d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
            for l in range(1, L+1):
                print("Layer ", l, ": ", enc_dec_layers[l-1], " -> ", enc_dec_layers[l])
                d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
                z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
                z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

                m, v = tf.nn.moments(z_pre_u, axes=[0])

                # if training:
                def training_batch_norm():
                    # Training batch normalization
                    # batch normalization for labeled and unlabeled examples is performed separately
                    if noise_std is not None:
                        # Corrupted encoder
                        # batch normalization + noise
                        z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                        z += tf.random_normal(tf.shape(z_pre)) * noise_std
                    else:
                        # Clean encoder
                        # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                        z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
                    return z

                # else:
                def eval_batch_norm():
                    # Evaluation batch normalization
                    # obtain average mean and variance and use it to normalize the batch
                    mean = ewma.average(running_mean[l-1])
                    var = ewma.average(running_var[l-1])
                    z = batch_normalization(z_pre, mean, var)
                    # Instead of the above statement, the use of the following 2 statements containing a typo
                    # consistently produces a 0.2% higher accuracy for unclear reasons.
                    # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                    # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
                    return z

                # perform batch normalization according to value of boolean "training" placeholder:
                z = tf.cond(self.training, training_batch_norm, eval_batch_norm)

                if l == L:
                    # use softmax activation in output layer
                    #h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
                    h = self.enc_final_lay(weights['gamma'][l-1], z, weights["beta"][l-1])
                else:
                    # use ReLU activation in hidden layers
                    h = tf.nn.relu(z + weights["beta"][l-1])
                d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
                d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
            d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
            return h, d

        print("=== Corrupted Encoder ===")
        h_corr, corr = encoder(X, noise_std)

        print("=== Clean Encoder ===")
        h_cln, clean = encoder(X, None)  # 0.0 -> do not add noise

        self.calc_preds(h_corr, h_cln)

        def g_gauss(z_c, u, size):
            "gaussian denoising function proposed in the original paper"
            wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
            a1 = wi(0., 'a1')
            a2 = wi(1., 'a2')
            a3 = wi(0., 'a3')
            a4 = wi(0., 'a4')
            a5 = wi(0., 'a5')

            a6 = wi(0., 'a6')
            a7 = wi(1., 'a7')
            a8 = wi(0., 'a8')
            a9 = wi(0., 'a9')
            a10 = wi(0., 'a10')

            mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
            v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

            z_est = (z_c - mu) * v + mu
            return z_est

        print("=== Decoder ===")
        # Decoder
        z_est = {}
        self.d_cost = []  # to store the denoising cost of all layers
        for l in range(L, -1, -1):
            print("Layer ", l, ": ", enc_dec_layers[l+1] if l+1 < len(enc_dec_layers) else None, " -> ", enc_dec_layers[l])#, ", denoising cost: ", denoising_cost[l])
            z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
            m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
            if l == L:
                u = unlabeled(h_corr)
            else:
                u = tf.matmul(z_est[l+1], weights['V'][l])
            u = batch_normalization(u)
            z_est[l] = g_gauss(z_c, u, enc_dec_layers[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            self.d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / enc_dec_layers[l]))

            
class BaseModel(Model):
    def __init__(self, enc_dec_layers, noise_std):
        self.enc_final_lay = lambda gamma, z, beta: tf.nn.softmax(gamma * (z + beta))
        super(BaseModel, self).__init__(enc_dec_layers, noise_std)
    
    def calc_preds(self, h_corr, h_cln):
        self.R_corr_L = self.labeled(h_corr)
        self.R_cln_L = self.labeled(h_cln)

        
class SetModel(Model):
    def __init__(self, enc_dec_layers, noise_std, sigma2):
        self.sigma2 = sigma2
        self.enc_final_lay = lambda gamma, z, beta: tf.identity(z)
        super(SetModel, self).__init__(enc_dec_layers, noise_std)

    def prediction(self, h_, labs):
        dists2 = pdist2(h_, self.labeled(h_))
        smax = tf.nn.softmax(-dists2/self.sigma2)
        labs_ = tf.matmul(smax, labs)
        return labs_

    def calc_preds(self, h_corr, h_cln):
        R_corr = self.prediction(h_corr, self.R_L)
        self.R_corr_L, R_corr_U = self.split_lu(R_corr)

        R_cln = self.prediction(h_cln, self.R_L)
        R_cln_L, self.R_cln_U = self.split_lu(R_cln)

    
class Optimizer(object):
    def __init__(self, model, denoising_cost, start_rate, decay_after):

        self.model = model
        smr_tr, smr_ts = [], []
        self.smr_tr, self.smr_ts = smr_tr, smr_ts
        self.logger = logging.getLogger('Optimizer')
        self.start_rate, self.decay_after = start_rate, decay_after

        with my_name_scope('classifier'):
            # calculate total unsupervised cost by adding the denoising cost of all layers
            u_cost = tf.add_n([c_d*lambda_l for c_d, lambda_l in zip(model.d_cost, denoising_cost[::-1])])
            #u_cost = tf.add_n(model.d_cost)

            cost = -tf.reduce_mean(tf.reduce_sum(model.R_L*tf.log(model.R_corr_L+1e-10), 1))  # supervised cost
            self.loss = cost + u_cost  # total cost
            smr_scl('loss', self.loss, smr_tr)

            #pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction

        with my_name_scope('training'):
            self.learning_rate = tf.Variable(start_rate, trainable=False, name='learning_rate')
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            smr_scl('learning_rate', self.learning_rate, smr_ts)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*model.bn_assigns)
        with tf.control_dependencies([train_step]):
            self.train_step = tf.group(bn_updates)
            
        self.init()

        self.summary_train_op = tf.summary.merge(smr_tr)
        self.summary_test_op = tf.summary.merge(smr_ts)

    def on_new_epoch(self, sess, last_epoch, num_epochs):
        if (last_epoch+1) >= self.decay_after:
            ratio = 1.0 * (num_epochs - (last_epoch+1))
            ratio = max(0, ratio / (num_epochs - self.decay_after))
            sess.run(self.learning_rate.assign(self.start_rate * ratio))

    def on_train(self, sess, add_summary, i, X_L, R_L, X_U):
        model = self.model
        feed_dict = {model.X_L: X_L, model.R_L: R_L, model.X_U: X_U, model.training: True}
        sess.run(self.train_step, feed_dict=feed_dict)

        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)


class BaseOptimizer(Optimizer):
    def init(self):
        with my_name_scope('error'):
            err = error_calc(self.model.R_L, self.model.R_cln_L)
            smr_scl('test', err, self.smr_ts)
            smr_scl('train', err, self.smr_tr)
            
    def on_test(self, sess, add_summary, i, X_L, R_L, X_T, R_T):
        model = self.model
        feed_dict = {model.X_L: X_T, model.R_L: R_T, model.X_U: X_T, model.training: False}
        add_summary(sess.run(self.summary_test_op, feed_dict=feed_dict), i)
    
    
class SetOptimizer(Optimizer):
    def init(self):
        with my_name_scope('error'):
            self.test_err = tf.placeholder(tf.float32)
            smr_scl('test', self.test_err, self.smr_ts)
            
    def on_test(self, sess, add_summary, i, X_L, R_L, X_T, R_T):
        feed_dict = {model.X_L: X_L, model.R_L: R_L, model.X_U: X_T, model.training: False}
        loss, R_cln_U__, learning_rate = sess.run([self.loss, model.R_cln_U, self.learning_rate], feed_dict=feed_dict)
        errr = 100*np.average(np.argmax(R_cln_U__,1) != np.argmax(R_T,1))
        add_summary(sess.run(self.summary_test_op, feed_dict={self.test_err:errr}), i)
        
    
class Trainer:
    def __init__(self, dataset):
        self.ds = dataset

    def train(self, sman, optimizer, num_epochs, batch_size):
        sess = sman.sess
        D_L, D_U, D_T = self.ds.train.labeled_ds, self.ds.train.unlabeled_ds, self.ds.test
        iters_per_epoch = int(D_U.num_examples/batch_size)
        iter_start, iter_end = iters_per_epoch*sman.last_epoch, iters_per_epoch*num_epochs

        for i in tqdm(range(iter_start, iter_end)):
            unlabeled_images, _ = D_U.next_batch(batch_size)
            optimizer.on_train(sess, sman.add_summary, i, D_L.images, D_L.labels, unlabeled_images)

            if i % iters_per_epoch == 0:
                last_epoch = int(i/iters_per_epoch)
                sman.save(last_epoch)
                optimizer.on_test(sess, sman.add_summary, i, D_L.images, D_L.labels, self.ds.test.images, self.ds.test.labels)
                optimizer.on_new_epoch(sess, last_epoch, num_epochs)


reset_all()
real_run = 1
new_run = 1
dset = 'digits' #digits/fashion
modt = 'set' #set/base
n_labeled = 1000
batch_size = 1000

if modt=='base':
    model = BaseModel(enc_dec_layers=[784, 1000, 500, 250, 250, 250, 10], noise_std=.3)
    optimizer = BaseOptimizer(model, denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10], start_rate=.02, decay_after=15)
    
if modt=='set':
    model = SetModel(enc_dec_layers=[784, 1000, 500, 250, 250, 250, 20], noise_std=.3, sigma2=1)
    optimizer = SetOptimizer(model, denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10], start_rate=.02, decay_after=15)
    
run_id = '%s_%s_%dn_labeled_%dbatch_size'%(dset, modt, n_labeled, batch_size)
sman = SessMan(run_id=run_id, new_run=new_run, real_run=real_run, cache_root=os.path.join('..', 'cache_ladder'))
trainer = Trainer(load_mnist(dset, n_labeled=n_labeled))

sman.load()
trainer.train(sman=sman, optimizer=optimizer, num_epochs=1000, batch_size=batch_size)
