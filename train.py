#!/opt/tensorflow/bin/python

from model import *
import input_data

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
