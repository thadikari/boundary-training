import tensorflow as tf
from tqdm import tqdm
import numpy as np
    
from boundary import build_boundary_set_ex
from common import *

def bi(inits, size):
    #return tf.Variable(inits * tf.ones([size]), name='b')
    with tf.variable_scope(tf.get_default_graph().get_name_scope(), reuse=tf.AUTO_REUSE):
        return tf.get_variable('var_b', size, initializer=tf.zeros_initializer)

def wi(shape):
    #init_val = np.random.normal(size=shape)*0.01
    #init_val = np.random.normal(size=shape)/math.sqrt(shape[0])
    #return tf.Variable(init_val, dtype='float', name='W')
    with tf.variable_scope(tf.get_default_graph().get_name_scope(), reuse=tf.AUTO_REUSE):
        return tf.get_variable('var_W', shape)

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
            
    log('Variables in graph: %d'%(len(tf.trainable_variables())))
    return last_actvn_fn(logits), logits, theta_lst

chkpts = [200, 600]

class BoundaryModel:
    def __init__(self, dim_x, dim_r, dim_t, layers, adv_train, actvn_fn, sigma, start_rate, regularizer, batch_size_bnd, epsilon_val, stop_grad):
    
        self.batch_size_bnd = batch_size_bnd
        self.epsilon_val = epsilon_val
        
        self.epsilon = tf.placeholder(tf.float32, shape=[], name='epsilon')
        self.X_L = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X_L')
        self.R_L = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R_L')
        
        self.X_B = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X_B')
        self.R_B = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R_B')
        
        num__L, num__B = tf.shape(self.X_L)[0], tf.shape(self.X_B)[0]
        __L, __B = lambda(dat): dat[:num__L], lambda(dat): dat[num__L:]
        
        def classifier(X_L, R_L, X_B, R_B):
            X = tf.concat([X_L, X_B], axis=0, name='X')
            R = tf.concat([R_L, R_B], axis=0, name='R')
            
            with my_name_scope('classifier'):
                T, T_logits, theta_T = create_fcnet(X, layers+[dim_t], tf.nn.relu, actvn_fn)
                #print(theta_T)
                
            with my_name_scope('projection'):
                T_L, T_B = __L(T), __B(T)
                dists2 = pdist2(T_L, T_B)
                smax = tf.nn.softmax(-dists2/sigma)
                R_hat_T = tf.matmul(smax, R_B)
                
            with my_name_scope('classifier'):
                cor = tf.clip_by_value(R_hat_T,1e-8,1.0) # self.R_hat_T + 1e-8 #
                print('make this addition')
                ttf = R_L * tf.log(cor)
                loss_label = -tf.reduce_mean(ttf)
                err = error_calc(R_L, R_hat_T)
                bsize = tf.shape(X_B)[0]
                
            return R_hat_T, loss_label, err, bsize, T_L, T_B

        R_hat_T, loss_label, self.err, bsize, T_L, T_B = classifier(self.X_L, self.R_L, self.X_B, self.R_B)
        grads_wrt_input = tf.gradients(loss_label, self.X_L)[0]
        peturb = self.epsilon*tf.sign(grads_wrt_input)
        X_L_tilde = tf.clip_by_value(self.X_L + peturb, 0., 1.)
        X_L_tilde = tf.stop_gradient(X_L_tilde) if stop_grad else X_L_tilde
        R_hat_T_tilde, loss_label_tilde, self.err_tilde, bsize_tilde, T_L_tilde, T_B_tilde = classifier(X_L_tilde, self.R_L, self.X_B, self.R_B)
        
        self.im_X, self.im_peturb, self.im_X_tilde = self.X_L, peturb, X_L_tilde
        T_L, T_B, T_L_tilde, R_hat_T_tilde, X_L_tilde, self.err_tilde, self.err = tf.identity(T_L, name='T_L'), tf.identity(T_B, name='T_B'), tf.identity(T_L_tilde, name='T_L_tilde'), tf.identity(R_hat_T_tilde, name='R_hat_T_tilde'), tf.identity(X_L_tilde, name='X_L_tilde'), tf.identity(self.err_tilde, name='err_tilde'), tf.identity(self.err, name='err')
        
        W2_ll = [tf.reduce_mean(tf.square(vv)) for vv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'var_W' in vv.name]
        loss_total = loss_label + (loss_label_tilde if adv_train else 0.) + regularizer*tf.add_n(W2_ll)
            
        # optimizing related code
        smr_tr, smr_ts = [], []
        with my_name_scope('testing'):
            smr_scl('error', self.err, smr_ts)
            smr_scl('error_tilde', self.err_tilde, smr_ts)
            smr_scl('bsize', bsize_tilde, smr_ts)

        with my_name_scope('training'):
            smr_scl('error', self.err, smr_tr)
            smr_scl('loss', loss_total, smr_tr)
            self.sub_list = []
            learning_rate = tf.Variable(start_rate, trainable=False)
            smr_scl('learning_rate', learning_rate, smr_ts)
            self.sub_list.append(RateUpdater(start_rate, learning_rate, chkpts))
            self.opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(loss=loss_total)
            
        self.summary_train_op = tf.summary.merge(smr_tr)
        self.summary_test_op = tf.summary.merge(smr_ts)
        
    def eval_trans(self, sess, X):
        return sess.run(tf.get_default_graph().get_tensor_by_name('T_L:0'), {self.X_L:X})
               
    def on_new_epoch(self, sess, last_epoch, num_epochs):
        for it in self.sub_list:
            it.on_new_epoch(sess, last_epoch, num_epochs)

    def update_set(self, sess, X, R):
        T = self.eval_trans(sess, X)
        _, pts = build_boundary_set_ex(T, R)
        self.bset = (X[pts], R[pts])

    def train_step(self, sess, add_summary, i, X, R):
        X_L, X_B, R_L, R_B = X, self.bset[0], R, self.bset[1]
        feed_dict = {self.X_L:X_L, self.X_B:X_B, self.R_L:R_L, self.R_B:R_B, self.epsilon:self.epsilon_val}
        sess.run(self.opt, feed_dict=feed_dict)
        
        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)

    def on_train(self, sess, add_summary, i, X, R):
        self.update_set(sess, X[:self.batch_size_bnd], R[:self.batch_size_bnd])
        self.train_step(sess, add_summary, i,
            X[self.batch_size_bnd:], R[self.batch_size_bnd:])

    def on_test(self, sess, add_summary, i, X, R):
        X_L, X_B, R_L, R_B = X, self.bset[0], R, self.bset[1]
        feed_dict = {self.X_L:X_L, self.X_B:X_B, self.R_L:R_L, self.R_B:R_B, self.epsilon:self.epsilon_val}
        add_summary(sess.run(self.summary_test_op, feed_dict=feed_dict), i)
        
                
class BaselineModel:
    def __init__(self, dim_x, dim_r, dim_t, layers, adv_train, start_rate, regularizer, epsilon_val, stop_grad):
        
        self.epsilon_val = epsilon_val
        
        # model related code
        self.epsilon = tf.placeholder(tf.float32, shape=[], name='epsilon')
        self.X = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X')
        self.R = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R')
        
        def classifier(X, R):
            with my_name_scope('classifier'):
                R_hat, R_hat_logits, theta_R_hat = create_fcnet(X, layers+[dim_t, dim_r], tf.nn.relu, tf.nn.softmax)
                # print theta_R_hat
                loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=R, logits=R_hat_logits))
                err = error_calc(R, R_hat)
                # smr_scl('loss', loss_label, smr_tr)
                return R_hat, loss_label, err
                
        R_hat, loss_label, self.err = classifier(self.X, self.R)
        grads_wrt_input = tf.gradients(loss_label, self.X)[0]
        peturb = self.epsilon*tf.sign(grads_wrt_input)
        X_tilde = tf.clip_by_value(self.X + peturb, 0., 1.)
        X_tilde = tf.stop_gradient(X_tilde) if stop_grad else X_tilde
        R_hat_tilde, loss_label_tilde, self.err_tilde = classifier(X_tilde, self.R)
        
        self.im_X, self.im_peturb, self.im_X_tilde = self.X, peturb, X_tilde
        
        W2_ll = [tf.reduce_mean(tf.square(vv)) for vv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'var_W' in vv.name]
        loss_total = loss_label + (loss_label_tilde if adv_train else 0.) + regularizer*tf.add_n(W2_ll)
            
        # optimizing related code
        smr_tr, smr_ts = [], []
        with my_name_scope('testing'):
            smr_scl('error', self.err, smr_ts)
            smr_scl('error_tilde', self.err_tilde, smr_ts)

        with my_name_scope('training'):
            smr_scl('error', self.err, smr_tr)
            smr_scl('loss', loss_total, smr_tr)
            self.sub_list = []
            learning_rate = tf.Variable(start_rate, trainable=False)
            smr_scl('learning_rate', learning_rate, smr_ts)
            self.sub_list.append(RateUpdater(start_rate, learning_rate, chkpts))
            self.opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(loss=loss_total)
            
        self.summary_train_op = tf.summary.merge(smr_tr)
        self.summary_test_op = tf.summary.merge(smr_ts)
        
    def on_new_epoch(self, sess, last_epoch, num_epochs):
        for it in self.sub_list:
            it.on_new_epoch(sess, last_epoch, num_epochs)
            
    def on_test(self, sess, add_summary, i, X, R):
        err, err_tilde, summary = sess.run([self.err, self.err_tilde, self.summary_test_op], feed_dict={self.X:X, self.R:R, self.epsilon:self.epsilon_val})
        add_summary(summary, i)
        print(err, err_tilde)
    
    def on_train(self, sess, add_summary, i, X, R):
        feed_dict = {self.X:X, self.R:R, self.epsilon:self.epsilon_val}
        sess.run(self.opt, feed_dict=feed_dict)
    
        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)


class ImageMan:
    def __init__(self, sman, model, D_T):
        
        self.model = model
        im_len = 28
        im_count = 10
        idx = random.sample(population=range(D_T.images.shape[0]), k=im_count)
        self.X_J, self.R_J = D_T.images[idx], D_T.labels[idx]#, np.argmax(D_T.labels, axis=1)[idx]
   
        def resh_(ims):
            nrows, ncols, height, width, intensity = (im_count, 1, im_len, im_len, 1)
            return tf.reshape(ims, [1, nrows, ncols, height, width, intensity])
            
        summary_test = []
        def gen_ims(ims):
            nrows, ncols, height, width, intensity = (im_count, 3, im_len, im_len, 1)
            ims = tf.reshape(ims, [1, nrows, ncols, height, width, intensity])
            ims = tf.transpose(ims, (0,1,3,2,4,5))
            ims = tf.reshape(ims, (1, height*nrows, width*ncols, intensity))
            summary_test.append(tf.summary.image('images', ims, max_outputs=20))

        peturb_im = .5 + model.im_peturb/2.
        ims = tf.concat([resh_(model.im_X), resh_(peturb_im), resh_(model.im_X_tilde)], axis=2)
        gen_ims(ims)
        #gen_ims('generated_images', model.X_hat)
        self.summary_test_op = tf.summary.merge(summary_test)
            
    def on_test(self, sess, add_summary, i, X, R):
        model = self.model
        add_summary(sess.run(self.summary_test_op, feed_dict={model.X_L: self.X_J, model.R_L: self.R_J, model.epsilon:model.epsilon_val}), i)

    def on_new_epoch(self, sess, last_epoch, num_epochs): pass
    def on_train(self, sess, add_summary, i, X, R): return

            
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
        
        
def make_model(modt, dim_t, start_rate, regularizer, epsilon_val, stop_grad, sigma, batch_size_bnd, adv_train, D_L):
    if modt=='set':
        actvn_fn = tf.identity
        return BoundaryModel(dim_x=784, dim_r=10, dim_t=dim_t, layers=[400,400], adv_train=adv_train, actvn_fn=actvn_fn, sigma=sigma, start_rate=start_rate, regularizer=regularizer, batch_size_bnd=batch_size_bnd, epsilon_val=epsilon_val, stop_grad=stop_grad)
        
    if modt=='baseline':
        return BaselineModel(dim_x=784, dim_r=10, dim_t=dim_t, layers=[400,400], adv_train=adv_train, start_rate=start_rate, regularizer=regularizer, epsilon_val=epsilon_val, stop_grad=stop_grad)
        
        
reset_all()
real_run = 1
new_run = 1

num_epochs = 1000
batch_size_bnd = 100
batch_size_trn = 100
dset = 'digits' #digits/fashion
modt = 'set' #set/baseline
start_rate = 0.001
regularizer = 0.001
epsilon_val = .25
adv_train = 1
stop_grad = 1
dim_t = 2
sigma = 60

run_id = '%s_%s_%dmbnd_%dmbtr_%ddim_t_%srate_%sregularizer_%sepsilon_val_%dsigma_%dadv_train_%dstop_grad'%(dset, modt, batch_size_bnd, batch_size_trn, dim_t, format_e(start_rate), format_e(regularizer), str(epsilon_val), sigma, adv_train, stop_grad)
trainer = Trainer(load_mnist(dset))
model = make_model(modt, dim_t, start_rate, regularizer, epsilon_val, stop_grad, sigma, batch_size_bnd, adv_train, trainer.ds.train.labeled_ds)
sman = SessMan(run_id=run_id, new_run=new_run, real_run=real_run)
imageman = ImageMan(sman, model, trainer.ds.test)
sman.load()
trainer.train(sman, modules=[model, imageman], num_epochs=num_epochs, batch_size=batch_size_bnd+batch_size_trn)
