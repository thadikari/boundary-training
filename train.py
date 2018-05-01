#!/opt/tensorflow/bin/python

from tqdm import tqdm
from model import *


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
        
reset_all()
real_run = 1
new_run = 1
num_epochs = 2000

trainer = Trainer(load_mnist_fashion())
model, optimizer = get_boundary_model(trainer.ds.train.labeled_ds)
sman = SessMan(run_id='fashion_boundary_2krun_1000MB_big_tree', new_run=new_run, real_run=real_run)
imageman = ImageMan(sman, model, trainer.ds.test)
sman.load()
trainer.train(sman, modules=[optimizer, imageman], num_epochs=num_epochs, batch_size=1000)
