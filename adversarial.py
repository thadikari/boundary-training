import tensorflow as tf
from tqdm import tqdm
import numpy as np
    
from boundary import build_boundary_set_ex, build_boundary_tree_ex, build_boundary_tree
from common import *

def bi(inits, size):
    #return tf.Variable(inits * tf.ones([size]), name='b')
    with tf.variable_scope(tf.get_default_graph().get_name_scope(), reuse=tf.AUTO_REUSE):
        return tf.get_variable('b', size, initializer=tf.zeros_initializer)

def wi(shape):
    #init_val = np.random.normal(size=shape)*0.01
    #init_val = np.random.normal(size=shape)/math.sqrt(shape[0])
    #return tf.Variable(init_val, dtype='float', name='W')
    with tf.variable_scope(tf.get_default_graph().get_name_scope(), reuse=tf.AUTO_REUSE):
        return tf.get_variable('W', shape)

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

                
class BaselineModel:
    def __init__(self, dim_x, dim_r, dim_t, layers, adv_train, start_rate):
        
        # model related code
        self.epsilon = tf.placeholder_with_default(.25, [], name='epsilon')
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
        self.peturb = self.epsilon*tf.sign(grads_wrt_input)
        self.X_tilde = tf.clip_by_value(self.X + self.peturb, 0., 1.)#tf.stop_gradient()
        R_hat_tilde, loss_label_tilde, self.err_tilde = classifier(self.X_tilde, self.R)
        
        loss_total = loss_label + (loss_label_tilde if adv_train else 0.)
            
        # optimizing related code
        smr_tr, smr_ts = [], []
        with my_name_scope('error'):
            smr_scl('train', self.err, smr_tr)
            smr_scl('test_err', self.err, smr_ts)
            smr_scl('test_err_tilde', self.err_tilde, smr_ts)

        with my_name_scope('training'):
            self.sub_list = []
            learning_rate = tf.Variable(start_rate, trainable=False, name='learning_rate')
            smr_scl('learning_rate', learning_rate, smr_ts)
            self.sub_list.append(RateUpdater(start_rate, learning_rate, [200, 600]))
            self.opt = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(loss=loss_total)
            
        self.summary_train_op = tf.summary.merge(smr_tr)
        self.summary_test_op = tf.summary.merge(smr_ts)
        
    def on_new_epoch(self, sess, last_epoch, num_epochs):
        for it in self.sub_list:
            it.on_new_epoch(sess, last_epoch, num_epochs)
            
    def on_test(self, sess, add_summary, i, X, R):
        err, err_tilde, summary = sess.run([self.err, self.err_tilde, self.summary_test_op], feed_dict={self.X:X, self.R:R})
        add_summary(summary, i)
        print(err, err_tilde)
    
    def on_train(self, sess, add_summary, i, X, R):
        feed_dict = {self.X:X, self.R:R}
        sess.run(self.opt, feed_dict=feed_dict)
    
        if i%500: add_summary(sess.run(self.summary_train_op, feed_dict=feed_dict), i)


class ImageMan:
    def __init__(self, sman, model, D_T):
        
        self.model = model
        im_len = 28
        im_count = 10
        idx = random.sample(population=range(D_T.images.shape[0]), k=im_count)
        self.X_B, self.R_B = D_T.images[idx], D_T.labels[idx]#, np.argmax(D_T.labels, axis=1)[idx]
   
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

        peturb_im = .5 + model.peturb/2.
        ims = tf.concat([resh_(model.X), resh_(peturb_im), resh_(model.X_tilde)], axis=2)
        gen_ims(ims)
        #gen_ims('generated_images', model.X_hat)
        self.summary_test_op = tf.summary.merge(summary_test)
            
    def on_test(self, sess, add_summary, i, X, R):
        model = self.model
        add_summary(sess.run(self.summary_test_op, feed_dict={model.X: self.X_B, model.R: self.R_B}), i)

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
        
        
def make_model(modt, start_rate, sigma, batch_size_bnd, batch_size_trn, adv_train, D_L):
    if modt=='set':
        actvn_fn = tf.identity
        model = BoundaryModel(dim_x=784, dim_r=10, dim_t=20, layers=[400,400], actvn_fn=actvn_fn, sigma=sigma)
        optimizer = SetOptimizer(model, start_rate, batch_size_bnd, batch_size_trn, D_L)
        return model, optimizer

    if modt=='baseline':
        return BaselineModel(dim_x=784, dim_r=10, dim_t=20, layers=[400,400], adv_train=adv_train, start_rate=start_rate)
        
        
reset_all()
real_run = 1
new_run = 1

num_epochs = 1000
batch_size_bnd = 100
batch_size_trn = 100
dset = 'digits' #digits/fashion
modt = 'baseline' #set/tree/tree_bch/baseline
start_rate = 0.001
adv_train = 0
sigma = 60

run_id = '%s_%s_%dmbnd_%dmbtr_%srate_%dsigma_%dadv_train'%(dset, modt, batch_size_bnd, batch_size_trn, format_e(start_rate), sigma, adv_train)
trainer = Trainer(load_mnist(dset))
model = make_model(modt, start_rate, sigma, batch_size_bnd, batch_size_trn, adv_train, trainer.ds.train.labeled_ds)
sman = SessMan(run_id=run_id, new_run=new_run, real_run=real_run)
imageman = ImageMan(sman, model, trainer.ds.test)
sman.load()
trainer.train(sman, modules=[model, imageman], num_epochs=num_epochs, batch_size=batch_size_bnd+batch_size_trn)
