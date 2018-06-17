import matplotlib.pyplot as plt
from sklearn import manifold
import tensorflow as tf
import pickle, sys, os
import numpy as np

from boundary import build_boundary_tree_ex, build_boundary_set_ex
from common import *


cache_root = os.path.join('..', 'cache')


def get_default_defs(run_id):
    if 'fashion' in run_id:
        dset = 'fashion'
        fontdict = {'fontsize':10, 'weight':"bold"}
        defs = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    elif 'digits' in run_id:
        dset = 'digits'
        defs = dict([(i, str(i)) for i in range(10)])
        fontdict = {'fontsize':20, 'weight':"bold"}
    return dset, fontdict, defs
        
            
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


def fn_time(run_id):
    
    #dset, fontdict, defs = get_default_defs(run_id)
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


def eval_dset__(cache_dir):
    ckpt = tf.train.get_checkpoint_state(cache_dir)
    assert(ckpt)
    print 'model_checkpoint: ', ckpt.model_checkpoint_path
    
    sess = tf.Session()
    saver = tf.train.import_meta_graph('%s.meta'%ckpt.model_checkpoint_path)
    tf.set_random_seed(0)
    saver.restore(sess, ckpt.model_checkpoint_path)
    last_epoch = int(ckpt.model_checkpoint_path.split('-')[-1])
    print('Restored epoch: %d'%last_epoch)
    return sess

def eval_dset(X_L, cache_dir):
    sess = eval_dset__(cache_dir)
    tf_T = tf.get_default_graph().get_tensor_by_name('classifier/Identity:0')
    tf_X_L = tf.get_default_graph().get_tensor_by_name('X_L:0')
    trans = sess.run(tf_T, {tf_X_L:X_L})
    return trans


def fn_trans(run_id):
    
    dset, fontdict, defs = get_default_defs(run_id)
    cache_dir = os.path.join(cache_root, run_id)
    cache_file = os.path.join(cache_dir, 'tsne_trans')
    if 0:#os.path.exists(cache_file):
        print 'cache_file exists:', cache_file
        trans, labels = pickle.load(open(cache_file,'r'))
    else:
        print 'no cache found'
        
        D_T = load_mnist(dset, n_labeled=10000).train.labeled_ds #digits/fashion
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
        plt.text(center[0], center[1], defs[i], fontdict=fontdict)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(os.path.join(cache_root, 'tsne.pdf'), bbox_inches='tight')
    plt.show()


def fn_gray(run_id):
    
    dset, fontdict, defs = get_default_defs(run_id)
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

    
get_tensor1 = lambda arg: tf.get_default_graph().get_tensor_by_name(arg+':0')
    
def eval_dset_ex(inps, outs, sess):
    feed_dict = {get_tensor1(kk):inps[kk] for kk in inps}
    outs_tt = [get_tensor1(kk) for kk in outs]
    return sess.run(outs_tt, feed_dict=feed_dict)

def permute(items):
    perm = np.arange(items[0].shape[0])
    np.random.shuffle(perm)
    return [tt[perm] for tt in items]

def hide_axticks(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.axis('off')

    
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import gridspec
import scipy.misc

def fn_ladder():
    cache_root = os.path.join('..', 'cache_ladder')
    run_id = '20180606-00:59:18_digits_set_1000n_labeled_1000batch_size_2dim_t'
    run_id = '20180606-01:13:41_digits_bndr_1000n_labeled_1000batch_size_2dim_t'
    reset_all(599544)

    if 'fashion' in run_id:
        dset = 'fashion'
        fontdict = {'fontsize':6, 'weight':'bold'}
        defs = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    elif 'digits' in run_id:
        dset = 'digits'
        defs = dict([(i, str(i)) for i in range(10)])
        fontdict = {'fontsize':12, 'weight':'bold'}

    cache_dir = os.path.join(cache_root, run_id)
    mnist = load_mnist(dset, n_labeled=10000)
    DS = mnist.test#train.labeled_ds
    X, R = permute([DS.images, DS.labels])

    sess = eval_dset__(cache_dir)
    T, Y = eval_dset_ex({'X_L':X}, ['T_L'], sess)[0], np.argmax(R, 1)
    assert(T.shape[1]==2)

    print 'plotting!'
    ax1 = plt.gca()

    #background scatter
    cmap = plt.get_cmap('tab10')
    ax1.scatter(T[:, 0], T[:, 1], marker='o', s=5, edgecolor='none', c=Y, cmap=cmap)

    #cluster center label
    for i in range(10):
        indices = Y == i
        center = np.average(T[indices], 0)
        ax1.text(center[0], center[1], defs[i], fontdict=fontdict)

    ax1.set_aspect('equal', adjustable='box')
    hide_axticks(ax1)
    plt.savefig(os.path.join(cache_root, '%s.pdf'%run_id), bbox_inches='tight')
    plt.show()
    print('saved plot.')


def fn_movie(run_id):
    reset_all(599544)
    dset, fontdict, defs = get_default_defs(run_id)
    
    baseline = 'baseline' in run_id
    if 'fashion' in run_id:
        dset = 'fashion'
        fontdict = {'fontsize':6, 'weight':'bold'}
        defs = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    elif 'digits' in run_id:
        dset = 'digits'
        defs = dict([(i, str(i)) for i in range(10)])
        fontdict = {'fontsize':10, 'weight':'bold'}

    cache_dir = os.path.join(cache_root, run_id)
    mnist = load_mnist(dset, n_labeled=10000)
    DS = mnist.train.labeled_ds
    X, R = permute([DS.images, DS.labels])
    sess = eval_dset__(cache_dir)
        
    if 0:
        T, Y = eval_dset_ex({'X_L':X}, ['T_L'], sess)[0], np.argmax(R, 1)
        _, pts = build_boundary_set_ex(T, R)
        pts = np.array(pts)
        X_B, R_B = X[pts], R[pts]
        print('boundary size: ', X_B.shape[0])

        rest = (X[~pts], R[~pts])
        X_L, R_L = rest[0][:5], rest[1][:5]
        print('points size: ', X_L.shape[0])
    else:
        ll_classes, from_each = range(10), 2
        #ll_classes, from_each = [6], 5
        X_B, R_B = X, R
        rest = (mnist.test.images, mnist.test.labels)
        inds = [it for ii in ll_classes for it in np.random.choice(np.nonzero(np.argmax(rest[1], 1)==ii)[0], from_each).tolist()]
        X_L, R_L = rest[0][inds], rest[1][inds]
        
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, 10))
   
    print 'plotting!'
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])
    ax1 = plt.subplot(gs[0])
    
    T_B = eval_dset_ex({('X' if baseline else 'X_B'):X_B}, [('T_logits' if baseline else 'T_B')], sess)[0]
    Y_L, Y_B = np.argmax(R_L, 1), np.argmax(R_B, 1)
    
    #background scatter
    assert(T_B.shape[1]==2)
    ax1.scatter(T_B[:, 0], T_B[:, 1], marker='o', s=5, edgecolor='none', c=Y_B, cmap=cmap)
    
    #generate path
    mossaic_ll = []
    for eps in [it*.01 for it in range(0, 30, 3)]:
        if baseline:
            feed_dict = {'X':X_L, 'R':R_L, 'epsilon':eps}
            out_list = ['T_logits_tilde', 'R_hat_tilde', 'X_tilde']
        else:
            feed_dict = {'X_L':X_L, 'R_L':R_L, 'X_B':X_B, 'R_B':R_B, 'epsilon':eps}
            out_list = ['T_L_tilde', 'R_hat_T_tilde', 'X_L_tilde']
        T_L_tilde, R_hat_T_tilde, X_L_tilde = eval_dset_ex(feed_dict, out_list, sess)
        #plt.scatter(T_L_tilde[:, 0], T_L_tilde[:, 1], marker='P', s=100, edgecolor='none', c='k')
        mossaic_ll.append(X_L_tilde.reshape([-1, 28, 28]).swapaxes(0,1).reshape(28, -1))
        for coords, preds, label, tilde in zip(T_L_tilde, R_hat_T_tilde, Y_L, X_L_tilde):
            if 1:
                #color = np.clip(np.sqrt(np.matmul(preds**2, colors)), 0., 1.)
                color = np.clip(np.matmul(preds, colors), 0., 1.)
                ax1.text(coords[0], coords[1], defs[label][0], fontdict={'fontsize':6, 'weight':'bold', 'color':color})
            else:
                img = np.clip(tilde, 0, 1).reshape([28,28])
                ofim = OffsetImage(img, zoom=1, cmap='gray')
                ab = AnnotationBbox(ofim, coords, xycoords='data', frameon=False)
                ax1.add_artist(ab)
                #ax.update_datalim(np.column_stack([x, y]))
                #ax.autoscale()

    #cluster center label
    for i in range(10):
        indices = Y_B == i
        center = np.average(T_B[indices], 0)
        ax1.text(center[0], center[1], defs[i], fontdict=fontdict)

    ax2 = plt.subplot(gs[1])
    mossaic = np.concatenate(mossaic_ll, 0)
    ax2.imshow(mossaic, cmap='gray')
    #scipy.misc.toimage(, cmin=0., cmax=1.).save(os.path.join(cache_root, 'outfile.pdf'))

    ax1.set_aspect('equal', adjustable='box')
    hide_axticks(ax1)
    hide_axticks(ax2)
    #plt.show()
    plt.savefig(os.path.join(cache_root, '%s.pdf'%run_id), bbox_inches='tight')
    print('saved plot.')
    
    # calc final test error using all traning data
    if not baseline:
        feed_dict = {'X_L':mnist.test.images, 'R_L':mnist.test.labels, 'X_B':mnist.train.labeled_ds.images, 'R_B':mnist.train.labeled_ds.labels, 'epsilon':.25}
        err, err_tilde = eval_dset_ex(feed_dict, ['err', 'err_tilde'], sess)
        print({'err':err, 'err_tilde':err_tilde})


def fn_movies(run_id):
    run_ll = [os.path.basename(x[0]) for x in os.walk(cache_root)]
    #return ([fn_movie(run_id) for run_id in run_ll if (('2dim_t' in run_id) and ('adv_train' in run_id) and ('_baseline_' not in run_id))])
    if run_id:
        return fn_movie(run_id)
    else:
        for run_id in [run_id for run_id in run_ll if '2dim_t' in run_id]:
            try:
                fn_movie(run_id)
            except:
                print('failed for [%s]'%run_id)
        
if __name__ == '__main__':
    
    reset_all(599)
    run_id = sys.argv[2] if len(sys.argv)>2 else None
    if sys.argv[1]=='time': fn_time(run_id)
    if sys.argv[1]=='trans': fn_trans(run_id)
    if sys.argv[1]=='gray': fn_gray(run_id)
    if sys.argv[1]=='movie': fn_movies(run_id)
    if sys.argv[1]=='ladder': fn_ladder(run_id)
