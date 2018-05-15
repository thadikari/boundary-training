import matplotlib.pyplot as plt
from sklearn import manifold
import tensorflow as tf
import pickle, sys, os
import numpy as np

from boundary import build_boundary_tree_ex
from common import *


cache_root = os.path.join('..', 'cache')


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
        plt.plot(tt, vv, label=label)

    ax.legend()
    plt.xscale('log')
    #plt.yscale('log')

    plt.xlabel('Wall clock time in log scale')
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.ylabel('Test erorr %')
    fig.savefig(os.path.join(cache_root, 'training_time_vert_fashion.pdf'), bbox_inches='tight')
    plt.show()


def eval_dset(X_L, cache_dir):
    ckpt = tf.train.get_checkpoint_state(cache_dir)
    assert(ckpt)
    print 'model_checkpoint: ', ckpt.model_checkpoint_path
    
    reset_all()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('%s.meta'%ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    last_epoch = int(ckpt.model_checkpoint_path.split('-')[-1])
    print('Restored epoch: %d'%last_epoch)

    tf_T = tf.get_default_graph().get_tensor_by_name('classifier/Identity:0')
    tf_X_L = tf.get_default_graph().get_tensor_by_name('X_L:0')
    trans = sess.run(tf_T, {tf_X_L:X_L})
    return trans


#run_id = '20180504-03:36:26_digits_set_1000mbnd_1000mbtr_1E-03rate_60sigma'
run_id = '20180504-03:34:02_fashion_set_100mbnd_100mbtr_1E-03rate_60sigma'
#run_id = '20180512-16:13:04_digits_set_100mbnd_100mbtr_1E-03rate_60sigma_0stop_grad_2d'
#run_id = '20180514-08:42:14_fashion_set_100mbnd_100mbtr_1E-03rate_60sigma_0stop_grad_2d'

dset = 'fashion' if 'fashion' in run_id else ('digits' if 'digits' in run_id else None)

def fn_trans():
    cache_dir = os.path.join(cache_root, run_id)
    cache_file = os.path.join(cache_dir, 'tsne_trans')
    if 0:#os.path.exists(cache_file):
        print 'cache_file exists:', cache_file
        trans, labels = pickle.load(open(cache_file,'r'))
    else:
        print 'no cache found'
        
        D_T = load_mnist(dset).test #digits/fashion
        T = eval_dset(D_T.images, cache_dir)
        labels = np.argmax(D_T.labels, 1)

        if T.shape[1]!=2:
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            trans = tsne.fit_transform(T)
        else:
            print 'no tsne needed'
            trans = T

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


def fn_gray():
    D_T = load_mnist(dset).train.labeled_ds #digits/fashion test#
    cache_dir = os.path.join(cache_root, run_id)
    T = eval_dset(D_T.images, cache_dir)
    
    print 'build_boundary_tree_ex'
    _, result = build_boundary_tree_ex(T, D_T.labels, D_T.images)
    isadded = np.array(result)
    print 'plotting!'

    nbdr = T[~isadded]
    labels = np.argmax(D_T.labels, 1)[isadded]
    trans = T[isadded]
    
    plt.scatter(nbdr[:, 0], nbdr[:, 1], marker='.', s=5, edgecolor='none', color = '0.75')
    plt.scatter(trans[:, 0], trans[:, 1], marker='o', s=20, edgecolor='none', c=labels, cmap=plt.get_cmap('tab10'))

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(os.path.join(cache_root, 'tsne.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    if sys.argv[1]=='time': fn_time()
    if sys.argv[1]=='trans': fn_trans()
    if sys.argv[1]=='gray': fn_gray()
