import matplotlib.pyplot as plt
from sklearn import manifold
import tensorflow as tf
import pickle, sys, os
import numpy as np
from model import *


def extract_events(path):
    tt, vv, ss = [], [], None
    try:
        for e in tf.train.summary_iterator(path):
            ss = ss if ss else e.wall_time
            for v in e.summary.value:
                if v.tag == 'error/test_final_BT':
                    tt.append(e.wall_time)
                    vv.append(v.simple_value)
    except Exception as e: print(e)
    #print np.array(tt), ss

    return np.array(tt)-ss, np.array(vv)


cache_root = os.path.join('..', 'cache')

def fn_time():
    lims = [(30,50000), (10,40)]
    runs = (['20180508-00:32:28_fashion_tree_1000mbnd_1000mbtr_1E-04rate_60sigma', 'DBT'],
            ['20180504-03:34:02_fashion_set_100mbnd_100mbtr_1E-03rate_60sigma', 'DBS'])

    #lims = [(110,35000), (1.5,20)]
    #runs = (['20180508-00:33:09_digits_tree_1000mbnd_1000mbtr_1E-04rate_60sigma', 'DBT'],
    #        ['20180503-04:23:41_digits_set_100mbnd_100mbtr_1E-03rate_60sigma', 'DBS'])

    data = []
    for run_id, label in runs:

        log_dir = os.path.join(cache_root, run_id)
        cache_file = os.path.join(log_dir, 'tmp_test_final_BT')

        if os.path.exists(cache_file):
            tt, vv = pickle.load(open(cache_file,'r'))
        else:
            for file in os.listdir(log_dir):
                if file.startswith('events.out.tfevents.'):
                    events_file = os.path.join(log_dir, file)
                    break

            print 'events_file:', events_file
            tt, vv = extract_events(events_file)
            pickle.dump([tt, vv], open(cache_file,'w'))

        data.append([tt, vv, label])

    fig = plt.figure(figsize=(2.6,5))
    ax = fig.add_subplot(111)

    for tt, vv, label in data:
        #tt = tt - np.min(tt) + 1
        plt.plot(tt, vv, label=label)
        #plt.yscale('log')

    ax.legend()
    plt.xscale('log')
    #plt.yscale('log')

    plt.xlabel('Wall clock time in log scale')
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.ylabel('Test erorr %')
    fig.savefig(os.path.join(cache_root, 'training_time_vert_fashion.pdf'), bbox_inches='tight')
    plt.show()


def fn_trans():
    #run_id = '20180504-03:36:26_digits_set_1000mbnd_1000mbtr_1E-03rate_60sigma'
    #run_id = '20180504-03:34:02_fashion_set_100mbnd_100mbtr_1E-03rate_60sigma'
    #run_id = '20180512-16:13:04_digits_set_100mbnd_100mbtr_1E-03rate_60sigma_0stop_grad_2d'
    run_id = '20180514-08:36:38_digits_set_100mbnd_100mbtr_1E-03rate_60sigma_0stop_grad_2d'

    cache_root = '../cache'
    cache_dir = os.path.join(cache_root, run_id)
    cache_file = os.path.join(cache_dir, 'tsne_trans')
    if os.path.exists(cache_file):
        print 'cache_file exists:', cache_file
        trans, labels = pickle.load(open(cache_file,'r'))
    else:
        print 'no cache found'
        reset_all()
        dset = 'digits' #digits/fashion
        modt = 'set' #set/tree/tree_bch/baseline
        stop_grad = 1
        sigma = 60

        model, optimizer = make_model(modt, 1, sigma, None, None, stop_grad, None)
        sess = tf.Session()
        saver = tf.train.Saver()
        #chckpt_path = '../checkpoints/checkpoints_%s/'%run_id
        ckpt = tf.train.get_checkpoint_state(cache_dir)
        if ckpt:
            print ckpt.model_checkpoint_path
            # if checkpoint exists, restore the parameters and set last_epoch and i_iter
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_epoch = int(ckpt.model_checkpoint_path.split('-')[-1])
            print('Restored epoch: %d'%last_epoch)

        D_T = load_mnist(dset).test
        T = model.eval_trans(sess, D_T.images)
        labels = np.argmax(D_T.labels, 1)

        if T.shape[1]!=2:
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            trans = tsne.fit_transform(T)
        else:
            print 'no tsne needed'
            trans = T
        #plt.ylabel(title)

        pickle.dump([trans, labels], open(cache_file,'w'))

    print 'plotting!'
    plt.scatter(trans[:, 0], trans[:, 1], marker='*', s=5, edgecolor='none', c=labels, cmap=plt.get_cmap('tab10'))
    for i in range(max(labels)+1):
        indices = labels == i
        center = np.average(trans[indices], 0)
        plt.text(center[0], center[1], str(i), fontsize=20, weight="bold")

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(os.path.join(cache_root, 'tsne.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    if sys.argv[1]=='time': fn_time()
    if sys.argv[1]=='trans': fn_trans()
