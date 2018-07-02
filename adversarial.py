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

def loss_with_spring(T_1, T_2, R_1, R_2):
    labels_t = tf.cast(tf.equal(tf.argmax(R_1,axis=1), tf.argmax(R_2,axis=1)), 'float')
    labels_f = 1. - labels_t
    eucd2 = tf.square(T_1 - T_2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")

    C = 5.0
    pos = labels_t * eucd2
    neg = labels_f * tf.square(tf.maximum(C - eucd, 0))
    losses = pos + neg
    return tf.reduce_mean(losses)


def loss_with_logs(T_1, T_2, R_1, R_2):
    labels_t = tf.cast(tf.equal(tf.argmax(R_1,axis=1), tf.argmax(R_2,axis=1)), 'float')
    labels_f = 1. - labels_t
    eucd2 = tf.square(T_1 - T_2)
    eucd2 = tf.reduce_sum(eucd2, 1)

    C = 10.
    probs = (1+tf.exp(-C))/(1+tf.exp(eucd2-C))
    ttf = labels_t * tf.log(tf.clip_by_value(probs,1e-8,1.0)) + labels_f * tf.log(tf.clip_by_value(1-probs,1e-8,1.0))
    return -tf.reduce_mean(ttf)

def gen_adv_ex(loss_term, inp_X, eps, name):
    grads_wrt_input = tf.gradients(loss_term, inp_X)[0]
    peturb = eps*tf.sign(grads_wrt_input)
    return tf.clip_by_value(inp_X + peturb, 0., 1., name=name)

    
class BoundaryModel:
    def __init__(self, dim_x, dim_r, dim_t, layers, adv_train, actvn_fn, sigma, start_rate, regularizer, batch_size_bnd, epsilon_val, siamese):
    
        self.siamese = siamese
        self.batch_size_bnd = batch_size_bnd
        self.epsilon_val = epsilon_val
        
        self.epsilon = tf.placeholder(tf.float32, shape=[], name='epsilon')
        self.X_L = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X_L')
        self.R_L = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R_L')
        
        self.X_B = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X_B')
        self.R_B = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R_B')
        
        num__L, num__B = tf.shape(self.X_L)[0], tf.shape(self.X_B)[0]
        __L, __B = lambda dat, nn: tf.identity(dat[:num__L], name=nn), lambda dat, nn: tf.identity(dat[num__L:], name=nn)
        
        def classifier(X_L, R_L, X_B, R_B, suffx):
            X = tf.concat([X_L, X_B], axis=0, name='X')
            R = tf.concat([R_L, R_B], axis=0, name='R')
            
            #with my_name_scope('classifier'):
            T, T_logits, theta_T = create_fcnet(X, layers+[dim_t], tf.nn.relu, actvn_fn)
            #print(theta_T)
                
            #with my_name_scope('projection'):
            T_L, T_B = __L(T, 'T_L'+suffx), __B(T, 'T_B'+suffx)
            dists2 = pdist2(T_L, T_B)
            smax = tf.nn.softmax(-dists2/sigma)
            R_hat_T = tf.matmul(smax, R_B, name='R_hat_T'+suffx)
            
            #with my_name_scope('classifier'):
            cor = tf.clip_by_value(R_hat_T,1e-8,1.0) # self.R_hat_T + 1e-8 #
            print('make this addition')
            ttf = R_L * tf.log(cor)
            loss_label = -tf.reduce_mean(ttf)
            err = error_calc(R_L, R_hat_T)
            bsize = tf.shape(X_B)[0]
            err = tf.identity(err, name='err'+suffx)
            loss_label = loss_with_spring(T_L, T_B, R_L, R_B) if self.siamese else loss_label
            return R_hat_T, loss_label, err, bsize, T_L, T_B
            #loss_with_spring

        R_hat_T, loss_label, self.err, bsize, T_L, T_B = classifier(self.X_L, self.R_L, self.X_B, self.R_B, '')

        X_L_tilde = gen_adv_ex(loss_label, self.X_L, self.epsilon, 'X_L_tilde')
        R_hat_T_tilde, loss_label_tilde, self.err_tilde, bsize_tilde, T_L_tilde, T_B_tilde = classifier(X_L_tilde, self.R_L, self.X_B, self.R_B, '_tilde')
        self.im_X, self.im_X_tilde = self.X_L, X_L_tilde
        
        W2_ll = [tf.reduce_mean(tf.square(vv)) for vv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'var_W' in vv.name]
        weight_loss = tf.add_n(W2_ll)
        
        if adv_train==1: adv_loss = loss_label_tilde
        elif adv_train==2: adv_loss = tf.reduce_mean(tf.square(tf.stop_gradient(T_L)-T_L_tilde))
        else: adv_loss = 0.
        
        adv_cog_loss = loss_label + adv_loss
        loss_total = adv_cog_loss + regularizer*weight_loss
            
        X_L_tilde2 = gen_adv_ex(adv_cog_loss, self.X_L, self.epsilon, 'X_L_tilde2')
        R_hat_T_tilde2, loss_label_tilde2, self.err_tilde2, bsize_tilde2, T_L_tilde2, T_B_tilde2 = classifier(X_L_tilde2, self.R_L, self.X_B, self.R_B, '_tilde2')

        self.im_X_tilde2 = X_L_tilde2
        # optimizing related code
        smr_tr, smr_ts = [], []
        with my_name_scope('testing'):
            smr_scl('error', self.err, smr_ts)
            if not self.siamese:
                smr_scl('error_tilde', self.err_tilde, smr_ts)
                smr_scl('error_tilde2', self.err_tilde2, smr_ts)
            smr_scl('bsize', bsize_tilde, smr_ts)

        with my_name_scope('training'):
            smr_scl('loss_label', loss_label, smr_tr)
            smr_scl('weight_loss', weight_loss, smr_tr)
            smr_scl('adv_loss', adv_loss, smr_tr)
            smr_scl('loss', loss_total, smr_tr)
            smr_scl('error', self.err, smr_tr)
            self.sub_list = []
            learning_rate = tf.Variable(start_rate, trainable=False)
            smr_scl('learning_rate', learning_rate, smr_ts)
            self.sub_list.append(RateUpdater(start_rate, learning_rate, chkpts))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss_total)
            #self.opt1 = tf.train.AdamOptimizer(learning_rate/10.).minimize(loss=adv_loss)
            
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

    def siamese_step(self, X, R):
        X_L, R_L = X[self.batch_size_bnd:], R[self.batch_size_bnd:]
        X_B, R_B = X[:self.batch_size_bnd], R[:self.batch_size_bnd]
        feed_dict = {self.X_L:X_L, self.X_B:X_B, self.R_L:R_L, self.R_B:R_B, self.epsilon:self.epsilon_val}
        return feed_dict

    def train_step(self, X, R):
        X_L, X_B, R_L, R_B = X, self.bset[0], R, self.bset[1]
        feed_dict = {self.X_L:X_L, self.X_B:X_B, self.R_L:R_L, self.R_B:R_B, self.epsilon:self.epsilon_val}
        return feed_dict

    def on_train(self, sess, add_summary, i, X, R):
        self.update_set(sess, X[:self.batch_size_bnd], R[:self.batch_size_bnd])
        if self.siamese:
            feed_dict = self.siamese_step(X, R)
        else:
            feed_dict = self.train_step(X[self.batch_size_bnd:], R[self.batch_size_bnd:])

        sess.run(self.opt, feed_dict=feed_dict)
        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)

    def on_test(self, sess, add_summary, i, X, R):
        X_L, X_B, R_L, R_B = X, self.bset[0], R, self.bset[1]
        feed_dict = {self.X_L:X_L, self.X_B:X_B, self.R_L:R_L, self.R_B:R_B, self.epsilon:self.epsilon_val}
        add_summary(sess.run(self.summary_test_op, feed_dict=feed_dict), i)
    
    def on_image_test(self, sess, add_summary, i, X, R, test_op):
        if self.siamese: return
        add_summary(sess.run(test_op, feed_dict={self.X_L: X, self.R_L: R, self.X_B:self.bset[0], self.R_B:self.bset[1], self.epsilon:self.epsilon_val}), i)
        
                
class BaselineModel:
    def __init__(self, dim_x, dim_r, dim_t, layers, adv_train, start_rate, regularizer, epsilon_val):
        
        self.epsilon_val = epsilon_val
        
        # model related code
        self.epsilon = tf.placeholder(tf.float32, shape=[], name='epsilon')
        self.X = tf.placeholder_with_default(tf.zeros([0,dim_x], tf.float32), shape=(None, dim_x), name='X')
        self.R = tf.placeholder_with_default(tf.zeros([0,dim_r], tf.float32), shape=(None, dim_r), name='R')
        
        def classifier(X, R, suffx):
            with my_name_scope('classifier'):
                T, T_logits, theta_T = create_fcnet(X, layers+[dim_t], tf.nn.relu, tf.nn.relu)
                T_logits = tf.identity(T_logits, name='T_logits'+suffx)
                R_hat, R_hat_logits, theta_R_hat = create_layer(T, dim_r, tf.nn.softmax)
                R_hat = tf.identity(R_hat, name='R_hat'+suffx)
                # print theta_R_hat
                loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=R, logits=R_hat_logits))
                err = error_calc(R, R_hat)
                # smr_scl('loss', loss_label, smr_tr)
                return R_hat, loss_label, err, T_logits
                
        R_hat, loss_label, self.err, T_logits = classifier(self.X, self.R, '')
        X_tilde = gen_adv_ex(loss_label, self.X, self.epsilon, 'X_tilde')
        R_hat_tilde, loss_label_tilde, self.err_tilde, T_logits_tilde = classifier(X_tilde, self.R, '_tilde')
        
        self.im_X, self.im_X_tilde = self.X, X_tilde
        
        W2_ll = [tf.reduce_mean(tf.square(vv)) for vv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'var_W' in vv.name]
            
        adv_loss = loss_label_tilde if adv_train else 0.
        adv_cog_loss = loss_label + adv_loss
        loss_total = loss_label + adv_cog_loss + regularizer*tf.add_n(W2_ll)
            
        X_tilde2 = gen_adv_ex(adv_cog_loss, self.X, self.epsilon, 'X_tilde2')
        R_hat_tilde2, loss_label_tilde2, self.err_tilde2, T_logits_tilde2 = classifier(X_tilde2, self.R, '_tilde2')
        self.im_X_tilde2 = X_tilde2

        # optimizing related code
        smr_tr, smr_ts = [], []
        with my_name_scope('testing'):
            smr_scl('error', self.err, smr_ts)
            smr_scl('error_tilde', self.err_tilde, smr_ts)
            smr_scl('error_tilde2', self.err_tilde2, smr_ts)

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
        #print(err, err_tilde)
    
    def on_train(self, sess, add_summary, i, X, R):
        feed_dict = {self.X:X, self.R:R, self.epsilon:self.epsilon_val}
        sess.run(self.opt, feed_dict=feed_dict)
    
        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)

    def on_image_test(self, sess, add_summary, i, X, R, test_op):
        model = self
        add_summary(sess.run(test_op, feed_dict={model.X: X, model.R: R, model.epsilon:model.epsilon_val}), i)
        

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
        def gen_ims(ims, len_ims):
            nrows, ncols, height, width, intensity = (im_count, len_ims, im_len, im_len, 1)
            ims = tf.reshape(ims, [1, nrows, ncols, height, width, intensity])
            ims = tf.transpose(ims, (0,1,3,2,4,5))
            ims = tf.reshape(ims, (1, height*nrows, width*ncols, intensity))
            summary_test.append(tf.summary.image('images', ims, max_outputs=20))

        #peturb_im = .5 + model.im_peturb/2.
        ll_ims = [resh_(model.im_X), resh_(model.im_X_tilde), resh_(model.im_X_tilde2), resh_(tf.abs(model.im_X_tilde-model.im_X_tilde2))]
        ims = tf.concat(ll_ims, axis=2)
        gen_ims(ims, len(ll_ims))
        #gen_ims('generated_images', model.X_hat)
        self.summary_test_op = tf.summary.merge(summary_test)
            
    def on_test(self, sess, add_summary, i, X, R):
        self.model.on_image_test(sess, add_summary, i, self.X_J, self.R_J, self.summary_test_op)

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
        
        
def make_model(modt, dim_t, start_rate, regularizer, epsilon_val, sigma, batch_size_bnd, adv_train, D_L, siamese):
    if modt=='set':
        actvn_fn = tf.identity
        return BoundaryModel(dim_x=784, dim_r=10, dim_t=dim_t, layers=[400,400], adv_train=adv_train, actvn_fn=actvn_fn, sigma=sigma, start_rate=start_rate, regularizer=regularizer, batch_size_bnd=batch_size_bnd, epsilon_val=epsilon_val, siamese=siamese)
        
    if modt=='baseline':
        return BaselineModel(dim_x=784, dim_r=10, dim_t=dim_t, layers=[400,400], adv_train=adv_train, start_rate=start_rate, regularizer=regularizer, epsilon_val=epsilon_val)
        
        
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
adv_train = 1 #1=standard FGSM / 2=nearest neigh
stop_grad = 1
siamese = 1
dim_t = 20
sigma = 60

run_id = '%s_%s_%dmbnd_%dmbtr_%ddim_t_%srate_%sregularizer_%sepsilon_val_%dsigma_%dadv_train_%dsiamese_newew_fixed_rerun'%(dset, modt, batch_size_bnd, batch_size_trn, dim_t, format_e(start_rate), format_e(regularizer), str(epsilon_val), sigma, adv_train, siamese)
trainer = Trainer(load_mnist(dset))
model = make_model(modt, dim_t, start_rate, regularizer, epsilon_val, sigma, batch_size_bnd, adv_train, trainer.ds.train.labeled_ds, siamese)
sman = SessMan(run_id=run_id, new_run=new_run, real_run=real_run)
imageman = ImageMan(sman, model, trainer.ds.test)
sman.load()
trainer.train(sman, modules=[model, imageman], num_epochs=num_epochs, batch_size=batch_size_bnd+batch_size_trn)
