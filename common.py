import time, os, random, scipy
import tensorflow as tf
import numpy as np

    
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

smr_scl = lambda name,opr,stp: stp.append(tf.summary.scalar(name,opr))
smr_hst = lambda name,opr,stp: None#stp.append(tf.summary.histogram(name,opr))

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


def reset_all(seed=0):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)

import input_data

def load_mnist(dset):
    if dset=='digits': return input_data.read_mnist('../data/digits', one_hot=True, SOURCE_URL=input_data.SOURCE_DIGITS)
    if dset=='fashion': return input_data.read_mnist('../data/fashion', one_hot=True, SOURCE_URL=input_data.SOURCE_FASHION)

time_id = lambda: time.strftime("%Y%m%d-%H:%M:%S", time.gmtime(time.mktime(time.gmtime())))
