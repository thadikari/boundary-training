#!/opt/tensorflow/bin/python

import tensorflow as tf
from tqdm import tqdm

import input_data
from model import Model, Optimizer, ImageMan, SessMan, reset_all


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
        
#for sigma in [45,55,60,70]:
reset_all()
real_run = 1
new_run = 1
num_epochs = 1000
actvn_fn = tf.identity
sigma = 1
btree_dists_in_ambient = 0

model = Model(dim_x=784, dim_r=10, dim_t=20,
              layers=[400,400], actvn_fn=actvn_fn, sigma=sigma
              )
optimizer = Optimizer(model, btree_dists_in_ambient=btree_dists_in_ambient)

sman = SessMan(run_id='actvn_%s_sigma_%d_benchmark'%(actvn_fn.__name__,1), new_run=new_run, real_run=real_run)
trainer = Trainer(input_data.read_mnist("../data", one_hot=True))
imageman = ImageMan(sman, model, trainer.ds.test)
sman.load()
trainer.train(sman, modules=[optimizer, imageman], num_epochs=num_epochs, batch_size=100)
