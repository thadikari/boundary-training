import tensorflow as tf
import numpy as np


def bi(inits, size):
    return tf.Variable(inits * tf.ones([size]), name='b')

def wi(shape):
    init_val = np.random.normal(size=shape)*0.01
    #init_val = np.random.normal(size=shape)/math.sqrt(shape[0])
    return tf.Variable(init_val, dtype='float', name='W')

def log(ss=''):
    nscp = tf.get_default_graph().get_name_scope()
    splt = nscp.split('/')
    print('\t'*(len(splt)-1) + splt[-1] + ': ' + ss)
    
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

get_dim1 = lambda vv: vv.get_shape().as_list()[1]

def create_layer__(acts_p, dim_l, actvn_fn):
    dim_lp = acts_p.get_shape().as_list()[1]
    W, b = wi((dim_lp, dim_l)), bi(0., dim_l)
    #print dim_lp, dim_l, W
    logits = tf.matmul(acts_p, W) + b
    return actvn_fn(logits), logits, [W, b]

def create_layer(acts_p, dim_l, actvn_fn):
    log('%d -> %d'%(get_dim1(acts_p), dim_l))
    return create_layer__(acts_p, dim_l, actvn_fn)

def create_fcnet(acts_p, layers, inner_actvn_fn, last_actvn_fn):
    log(' -> '.join([str(get_dim1(acts_p))] + [str(it) for it in layers]))
    theta_lst = []
    for l in range(len(layers)):
        with my_name_scope('layer_%d'%(l+1)):
            acts, logits, theta = create_layer__(acts_p, layers[l], inner_actvn_fn)
            theta_lst.extend(theta)
            acts_p = acts
            
    return last_actvn_fn(logits), logits, theta_lst

    
def pdist2(X, Y=None): # dimensions should be, X: NX x C and Y: NY x C
    # X2 = sum(X.^2,1); U = repmat(X2,N,1) + repmat(X2',1,N) - 2*(X'*X);
    X2 = tf.reduce_sum(tf.square(X),1)
    Y2 = X2 if Y is None else tf.reduce_sum(tf.square(Y),1)
    X_ = tf.expand_dims(X2, 1)
    Y_ = tf.expand_dims(Y2, 0)
    NX, NY = tf.shape(X2)[0], tf.shape(Y2)[0]
    X_T = tf.tile(X_, [1, NY])
    Y_T = tf.tile(Y_, [NX, 1])
    dists2 = X_T + Y_T - 2 * tf.matmul(X,tf.transpose(X if Y is None else Y))
    return dists2
    #return tf.sqrt(dists2 + 1e-10)
    

class my_name_scope(object):
    def __init__(self, name):
        cur = tf.get_default_graph().get_name_scope()
        self.ns = tf.name_scope(cur + ('' if len(cur)==0 else '/') + name + '/')
    def __enter__(self):
        self.ns.__enter__()
        return self.ns
    def __exit__(self, type, value, traceback):
        self.ns.__exit__(type, value, traceback)
    
    
class BoundaryModel:
    def __init__(self, dim_x, dim_r, dim_t, layers, actvn_fn, sigma):
    
        self.dim_x = dim_x
        self.X_L = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X_L')
        self.R_L = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R_L')
        
        self.X_B = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X_B')
        self.R_B = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R_B')
        
        self.X = tf.concat([self.X_L, self.X_B], axis=0, name='X')
        self.R = tf.concat([self.R_L, self.R_B], axis=0, name='R')
        
        num__L, num__B = tf.shape(self.X_L)[0], tf.shape(self.X_B)[0]
        __L, __B = lambda(dat): dat[:num__L], lambda(dat): dat[num__L:]
        
        with my_name_scope('classifier'):
            self.T, self.T_logits, self.theta_T = create_fcnet(self.X, layers+[dim_t], tf.nn.relu, actvn_fn)
            
        with my_name_scope('projection'):
            T_L, T_B = __L(self.T), __B(self.T)
            dists2 = pdist2(T_L, T_B)
            smax = tf.nn.softmax(-dists2/sigma)
            self.R_hat_T = tf.matmul(smax, self.R_B)
            
    def eval_trans(self, sess, X):
        return sess.run(self.T, {self.X_L:X})
        
        
class RateUpdater:
    def __init__(self, start_rate, rate_var):
        self.ll = [400, 1000, 3000]
        self.look_for = self.ll.pop(0)
        self.curr_val = start_rate
        self.rate_var = rate_var

    def on_new_epoch(self, sess, last_epoch, num_epochs):
        if self.look_for == last_epoch:
            self.curr_val *= .1
            sess.run(self.rate_var.assign(self.curr_val))
            self.look_for = self.ll.pop(0) if self.ll else -1
            #print last_epoch, self.rate_var.name, self.curr_val
        

from boundary import build_boundary_set_ex, build_boundary_tree
import logging

smr_scl = lambda name,opr,stp: stp.append(tf.summary.scalar(name,opr))
smr_hst = lambda name,opr,stp: None#stp.append(tf.summary.histogram(name,opr))

def error_calc(real_labels, pred_logits):
    not_eql = tf.not_equal(tf.argmax(real_labels,axis=1), tf.argmax(pred_logits,axis=1))
    return 100.*tf.reduce_mean(tf.cast(not_eql, 'float'))

def calc_BT_err(btree, T, R):
    err = 0.
    for tt, rr in zip(T, R):
        err += (np.argmax(btree.infer_probs(tt, 1)) != np.argmax(rr))
    test_error = 100.*err/T.shape[0]
    return test_error

    
class BoundaryOptimizer:
    def __init__(self, model, start_rate, batch_size_bnd, batch_size_trn, D_L):
        self.model = model
        self.D_L = D_L
        self.logger = logging.getLogger('Optimizer')
        self.smr_tr, self.smr_ts = [], []
        self.batch_size_bnd = batch_size_bnd
        self.batch_size_trn = batch_size_trn
        
        with my_name_scope('classifier'):
            cor = tf.clip_by_value(model.R_hat_T,1e-8,1.0) # self.R_hat_T + 1e-8 #
            ttf = model.R_L * tf.log(cor)
            loss_label = -tf.reduce_mean(ttf)
            smr_scl('loss', loss_label, self.smr_tr)

        self.init()
        
        with my_name_scope('error'):
            self.test_error_final_BT = tf.placeholder(tf.float32)
            smr_scl('test_final_BT', self.test_error_final_BT, self.smr_ts)

        with my_name_scope('boundary_size'):
            self.size_final_BT = tf.placeholder(tf.float32)
            smr_scl('final_BT', self.size_final_BT, self.smr_ts)

        with my_name_scope('training'):
            self.sub_list = []
            learning_rate = tf.Variable(start_rate, trainable=False, name='learning_rate')
            smr_scl('learning_rate', learning_rate, self.smr_ts)
            self.sub_list.append(RateUpdater(start_rate, learning_rate))
            self.opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(loss=loss_label, var_list=model.theta_T)
            
        self.summary_train_op = tf.summary.merge(self.smr_tr)
        self.summary_test_op = tf.summary.merge(self.smr_ts)
        
    def on_new_epoch(self, sess, last_epoch, num_epochs):
        for it in self.sub_list:
            it.on_new_epoch(sess, last_epoch, num_epochs)

    def build_final_BT(self, sess):
        D_L = self.D_L
        perm = np.arange(D_L._num_examples)
        np.random.shuffle(perm)
        X_L, R_L = D_L.images[perm], D_L.labels[perm]
        T_L = self.model.eval_trans(sess, X_L)
        return build_boundary_tree(T_L, R_L, X_L)

    def eval_final_BT(self, sess, T, R):
        final_BT = self.build_final_BT(sess)
        test_error_final_BT = calc_BT_err(final_BT, T, R)
        return test_error_final_BT, final_BT.size


class SetOptimizer(BoundaryOptimizer):
    def init(self):
        self.bset = None
        model, smr_tr, smr_ts = self.model, self.smr_tr, self.smr_ts
        
        with my_name_scope('error'):
            err = error_calc(model.R_L, model.R_hat_T)
            smr_scl('train', err, smr_tr)
            smr_scl('test_mini_batch', err, smr_ts)
            
        with my_name_scope('boundary_size'):
            self.size_training = tf.shape(model.X_B)[0]
            smr_scl('training', self.size_training, smr_tr)
            
    def on_test(self, sess, add_summary, i, X, R):
        
        if np.random.uniform()>.1:return # since final BT build/test are costly
        
        model = self.model
        T = self.model.eval_trans(sess, X)
        test_error_final_BT, size_final_BT = self.eval_final_BT(sess, T, R)
        X_L, X_B, R_L, R_B = X, self.bset[0], R, self.bset[1]
        feed_dict = {model.X_L:X_L, model.X_B:X_B, model.R_L:R_L, model.R_B:R_B, self.test_error_final_BT:test_error_final_BT, self.size_final_BT:size_final_BT}
        summary = sess.run(self.summary_test_op, feed_dict=feed_dict)
        
        self.logger.info(str({'test_error_final_BT':test_error_final_BT, 'size_final_BT':size_final_BT}))
        add_summary(summary, i)
        
    def update_set(self, sess, X, R):
        model = self.model
        T = model.eval_trans(sess, X)
        bset, pts = build_boundary_set_ex(T, R)
        self.bset = (X[pts], R[pts])

    def train_step(self, sess, add_summary, i, X, R):
        model = self.model
        X_L, X_B, R_L, R_B = X, self.bset[0], R, self.bset[1]
        feed_dict = {model.X_L:X_L, model.X_B:X_B, model.R_L:R_L, model.R_B:R_B}
        sess.run(self.opt, feed_dict=feed_dict)
    
        if i%50: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)

    def on_train(self, sess, add_summary, i, X, R):
        self.update_set(sess, X[:self.batch_size_bnd], R[:self.batch_size_bnd])
        tot = self.batch_size_bnd+self.batch_size_trn
        self.train_step(sess, add_summary, i,
            X[self.batch_size_bnd:tot], R[self.batch_size_bnd:tot])

        
class TreeOptimizer(BoundaryOptimizer):
    def init(self):
        self.btree = None
        model, smr_tr, smr_ts = self.model, self.smr_tr, self.smr_ts
    
        with my_name_scope('error'):
            self.test_error_MB = tf.placeholder(tf.float32)
            smr_scl('test_mini_batch', self.test_error_MB, smr_ts)

        with my_name_scope('boundary_size'):
            self.size_training = tf.placeholder(tf.float32)
            smr_scl('training', self.size_training, smr_tr)

    def on_test(self, sess, add_summary, i, X, R):
        model = self.model
        T = self.model.eval_trans(sess, X)
        test_error_final_BT, size_final_BT = self.eval_final_BT(sess, T, R)
        test_error_MB = calc_BT_err(self.btree, T, R)
        summary = sess.run(self.summary_test_op, feed_dict={self.test_error_MB:test_error_MB, self.test_error_final_BT:test_error_final_BT, self.size_final_BT:size_final_BT, self.size_training:self.btree.size})
        
        self.logger.info(str({'test_error_MB':test_error_MB, 'test_error_final_BT':test_error_final_BT, 'size_training':self.btree.size, 'size_final_BT':size_final_BT}))
        add_summary(summary, i)
        
    def update_set(self, sess, T, X, R):
        self.btree = build_boundary_tree(T, R, X)
    
    def train_step(self, sess, add_summary, i, T, X, R):
        model = self.model
        for ind in range(X.shape[0]):
            X_L, R_L = X[[ind],:], R[[ind],:]
            X_B, R_B = self.btree.query_neighbors(T[ind])
            feed_dict = {model.X_L:X_L, model.X_B:X_B, model.R_L:R_L, model.R_B:R_B, self.size_training:self.btree.size}
            sess.run(self.opt, feed_dict=feed_dict)
        
        if i%50: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)
    
    def on_train(self, sess, add_summary, i, X, R):
        model = self.model
        T = sess.run(model.T, {model.X_L:X})
        self.update_set(sess, T[:self.batch_size_bnd], X[:self.batch_size_bnd], R[:self.batch_size_bnd])
        tot = self.batch_size_bnd+self.batch_size_trn
        self.train_step(sess, add_summary, i,
            T[self.batch_size_bnd:tot], X[self.batch_size_bnd:tot],
            R[self.batch_size_bnd:tot])

                
class BaselineModel:
    def __init__(self, dim_x, dim_r, dim_t, layers):

        self.X = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X')
        self.R = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R')
        
        with my_name_scope('classifier'):
            self.R_hat, self.R_hat_logits, self.theta_R_hat = create_fcnet(self.X, layers+[dim_t, dim_r], tf.nn.relu, tf.nn.softmax)
            
        self.T = self.R_hat_logits
            
                
class BaselineOptimizer:
    def __init__(self, model):
        
        self.model = model
        smr_tr, smr_ts = [], []
        
        with my_name_scope('classifier/R_hat_T'):
            loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model.R, logits=model.R_hat_logits))
            smr_scl('loss', loss_label, smr_tr)

            with my_name_scope('error'):
                err = error_calc(model.R, model.R_hat)
                smr_scl('train', err, smr_tr)
                smr_scl('test', err, smr_ts)

        with my_name_scope('training'):
            self.sub_list = []
            s_rate = .001
            learning_rate = tf.Variable(s_rate, trainable=False, name='learning_rate')
            smr_scl('learning_rate', learning_rate, smr_ts)
            self.sub_list.append(RateUpdater(s_rate, learning_rate))
            self.opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(loss=loss_label, var_list=model.theta_R_hat)
            
        self.summary_train_op = tf.summary.merge(smr_tr)
        self.summary_test_op = tf.summary.merge(smr_ts)
        
    def on_new_epoch(self, sess, last_epoch, num_epochs):
        for it in self.sub_list:
            it.on_new_epoch(sess, last_epoch, num_epochs)
            
    def on_test(self, sess, add_summary, i, X, R):
        model = self.model
        summary = sess.run(self.summary_test_op, feed_dict={model.X:X, model.R:R})
        add_summary(summary, i)
    
    def on_train(self, sess, add_summary, i, X, R):
        model = self.model
        feed_dict = {model.X:X, model.R:R}
        sess.run(self.opt, feed_dict=feed_dict)
    
        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)

        
import time, os, random, scipy


class ImageMan:
    def __init__(self, sman, model, D_T):
        
        X_T, Y_T = D_T.images, np.argmax(D_T.labels, axis=1)
        im_len = 28

        ## displaying embedding
        embd_side = 32
        embd_count = embd_side**2
        idx = np.random.randint(D_T.num_examples, size=embd_count)
        self.X_E, self.Y_E = X_T[idx,:], Y_T[idx]
        self.embd_var = tf.Variable(tf.zeros([embd_count, model.T.get_shape().as_list()[1]]), name="embedding", trainable=False)
        self.assignment = self.embd_var.assign(model.T)

        LABELS, SPRITES = 'labels_%d.tsv'%embd_count, 'sprite_%d.png'%embd_count
        #SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = self.embd_var.name
        embedding_config.sprite.image_path = SPRITES
        embedding_config.metadata_path = LABELS
        embedding_config.sprite.single_image_dim.extend([im_len, im_len])

        if sman.cache_dir:
            with tf.summary.FileWriter(sman.cache_dir) as writer:
                tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

            lab_path = os.path.join(sman.cache_dir,'labels_%d.tsv'%embd_count)
            np.savetxt(lab_path, self.Y_E, fmt='%d', delimiter=',')

            spr_path = os.path.join(sman.cache_dir,'sprite_%d.png'%embd_count)
            spr_im = self.X_E.reshape(embd_side, embd_side, im_len, im_len).swapaxes(1,2).reshape(embd_side*im_len, embd_side*im_len)
            scipy.misc.imsave(spr_path, 1-spr_im)


        ## displaying images
        num_patterns = 10
        num_from_each = 1
        self.model = model
        B_T = np.empty((0,784))
        R_T = np.empty((0,10))
        for ll in range(10):
            off = 400
            tmp = X_T[Y_T==ll][off:off+num_from_each]
            B_T = np.vstack([B_T, np.tile(tmp,(num_patterns,1))])
            R_T = np.vstack([R_T, np.eye(10)[:num_patterns]])
        self.B_T = B_T
        self.R_T = R_T
                
        summary_test = []
        def gen_ims(name, ims):
            nrows, ncols, height, width, intensity = (10, num_patterns, im_len, im_len, 1)
            ims = tf.reshape(ims, [1, nrows, ncols, height, width, intensity])
            ims = tf.transpose(ims, (0,1,3,2,4,5))
            ims = tf.reshape(ims, (1, height*nrows, width*ncols, intensity))
            summary_test.append(tf.summary.image(name, ims, max_outputs=20))

        #gen_ims('original_images', model.X)
        #gen_ims('generated_images', model.X_hat)
        #self.summary_test_op = tf.summary.merge(summary_test)
            
    def on_test(self, sess, add_summary, i, X, R):
        model = self.model
        sess.run(self.assignment, feed_dict={model.X: self.X_E})
        return
        add_summary(sess.run(self.summary_test_op, feed_dict={model.X_L: self.B_T, model.R_L: self.R_T}), i)

    def on_new_epoch(self, sess, last_epoch, num_epochs): pass
    def on_train(self, sess, add_summary, i, X, R): return

            
time_id = lambda: time.strftime("%Y%m%d-%H:%M:%S", time.gmtime(time.mktime(time.gmtime())))


class SessMan:
    def __init__(self, run_id, new_run, real_run):

        cache_root = os.path.join('..', 'cache')
        def mkdir():
            new_dir = os.path.join(cache_root, '%s_%s'%(time_id(),run_id))
            if not os.path.exists(new_dir): os.makedirs(new_dir)
            return new_dir
        
        def load_cached(cache_dir):
            self.ckpt = tf.train.get_checkpoint_state(cache_dir)  # get latest checkpoint (if any)
            if self.ckpt and self.ckpt.model_checkpoint_path: # should continue from this checkpoint
                print('Loaded checkpoint. Caching in EXISTING dir: %s'%cache_dir)
            else:
                #cache_dir = mkdir()
                print('No checkpoints found. Caching in EXISTING dir: %s'%cache_dir)
                self.ckpt = None

        self.real_run = real_run
        self.cache_dir = None
        self.ckpt = None
        
        if not self.real_run:
            print('*********NOT A REAL RUN!')
            return
        
        if new_run:
            cache_dir = mkdir()
            print('Starting NEW run. Caching in NEW dir: %s'%cache_dir)
        else: # continue form last run if checkpoint exists
            if len(os.listdir(cache_root))==0: # cache dir empty
                cache_dir = mkdir()
                print('No runs found. Starting NEW run. Caching in NEW dir: %s'%cache_dir)
            else: # get the last updated dir
                cache_dir = max([os.path.join(cache_root,d) for d in os.listdir(cache_root)], key=os.path.getmtime)
                print('Attempting to load from last updated dir: %s'%cache_dir)
                load_cached(cache_dir)
                
        self.cache_dir = cache_dir

        logging.basicConfig(filename=os.path.join(self.cache_dir,'log.log'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logging.info('initialized logger to file, PID:[%s]'%str(os.getpid()))


    def load(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        #self.chckpt_path = '../checkpoints/checkpoints_%s/'%run_id
        if self.ckpt:
            # if checkpoint exists, restore the parameters and set self.last_epoch and i_iter
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.last_epoch = int(self.ckpt.model_checkpoint_path.split('-')[-1])
            print('Restored epoch: %d'%self.last_epoch)
        else:
            print('New run from epoch 0.')
            self.sess.run(tf.global_variables_initializer())
            self.last_epoch = 0

        if self.real_run:
            self.writer = tf.summary.FileWriter(self.cache_dir, graph=tf.get_default_graph())
            import glob, shutil
            for file in glob.glob(os.path.join(os.path.dirname(__file__),'*.py')):
                shutil.copy(file, self.cache_dir)

    def add_summary(self, summary, i):
        if self.real_run:
            self.writer.add_summary(summary, i)

    def save(self, epoch):
        if self.real_run:
            self.saver.save(self.sess, os.path.join(self.cache_dir, 'model.ckpt'), epoch)

            
from tqdm import tqdm

class Trainer:
    def __init__(self, dataset):
        self.ds = dataset
        
    def train(self, sman, modules, num_epochs, batch_size):
        sess = sman.sess
        last_epoch = sman.last_epoch
        
        iters_per_epoch = int(self.ds.train.unlabeled_ds.num_examples/batch_size)
        iter_start, iter_end = iters_per_epoch*last_epoch, iters_per_epoch*num_epochs
        
        for i in tqdm(range(iter_start, iter_end)):
            for module in modules:
                module.on_train(sess, sman.add_summary, i, *self.ds.train.labeled_ds.next_batch(batch_size))

            if i % iters_per_epoch == 0:
                last_epoch = int(i/iters_per_epoch)
                sman.save(last_epoch)
                for module in modules:
                    module.on_test(sess, sman.add_summary, i, self.ds.test.images, self.ds.test.labels)
                for module in modules:
                    module.on_new_epoch(sess, last_epoch, num_epochs)
        
        
def reset_all(seed=0):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)

import input_data

def load_mnist(dset):
    if dset=='digits': return input_data.read_mnist('../data/digits', one_hot=True, SOURCE_URL=input_data.SOURCE_DIGITS)
    if dset=='fashion': return input_data.read_mnist('../data/fashion', one_hot=True, SOURCE_URL=input_data.SOURCE_FASHION)

def make_model(modt, start_rate, sigma, batch_size_bnd, batch_size_trn, D_L):
    if modt=='set':
        actvn_fn = tf.identity
        #sigma = 60, start_rate = 0.001
        model = BoundaryModel(dim_x=784, dim_r=10, dim_t=20, layers=[400,400], actvn_fn=actvn_fn, sigma=sigma)
        optimizer = SetOptimizer(model, start_rate, batch_size_bnd, batch_size_trn, D_L)
        return model, optimizer

    if modt=='tree':
        actvn_fn = tf.identity
        #sigma = 60, start_rate = 0.0001
        model = BoundaryModel(dim_x=784, dim_r=10, dim_t=20, layers=[400,400], actvn_fn=actvn_fn, sigma=sigma)
        optimizer = TreeOptimizer(model, start_rate, batch_size_bnd, batch_size_trn, D_L)
        return model, optimizer

    if modt=='baseline':
        model = BaselineModel(dim_x=784, dim_r=10, dim_t=20, layers=[400,400])
        optimizer = BaselineOptimizer(model)
        return model, optimizer
