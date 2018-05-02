#!/opt/tensorflow/bin/python

from model import *

reset_all()
real_run = 1
new_run = 1

num_epochs = 2000
batch_size = 1000
dset = 'fashion' #digits/fashion
modt = 'set' #set/tree/baseline

run_id = '%s_%s_%depochs_%dmb__dududud'%(dset, modt, num_epochs, batch_size)
trainer = Trainer(load_mnist(dset))
model, optimizer = make_model(modt, trainer.ds.train.labeled_ds)
sman = SessMan(run_id=run_id, new_run=new_run, real_run=real_run)
#imageman = ImageMan(sman, model, trainer.ds.test)
sman.load()
trainer.train(sman, modules=[optimizer], num_epochs=num_epochs, batch_size=batch_size)
