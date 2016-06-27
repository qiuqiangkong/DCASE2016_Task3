'''
SUMMARY:  detect each wav from cross validation
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: -
--------------------------------------
'''
import sys
sys.path.append( '/homes/qkong/my_code2015.5-/python/Hat' )
sys.path.append( 'evaluation' )
import numpy as np
import config as cfg
import prepareData as ppData
from Hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Dropout, Flatten
from Hat.callbacks import SaveModel, Validation
from Hat.optimizers import SGD, Rmsprop
import Hat.backend as K
import os
import cPickle
import pickle
import matplotlib.pyplot as plt
from evaluation import *

# hyper-params
fold = 1        # can be 1,2,3,4
type = 'home'   # can be 'home' or 'resi'
agg_num = 10
hop = 1
thres = 0.99

# init paths
if type=='home':
    fe_fd = cfg.fe_mel_home_fd
    labels = cfg.labels_home
    lb_to_id = cfg.lb_to_id_home
    id_to_lb = cfg.id_to_lb_home
    tr_txt = cfg.evaluation_fd + '/home_fold' + str(fold) + '_train.txt'
    te_txt = cfg.evaluation_fd + '/home_fold' + str(fold) + '_evaluate.txt'
    meta_fd = cfg.meta_home_fd
if type=='resi':
    fe_fd = cfg.fe_mel_resi_fd
    labels = cfg.labels_resi
    lb_to_id = cfg.lb_to_id_resi
    id_to_lb = cfg.id_to_lb_resi
    tr_txt = cfg.evaluation_fd + '/residential_area_fold' + str(fold) + '_train.txt'
    te_txt = cfg.evaluation_fd + '/residential_area_fold' + str(fold) + '_evaluate.txt'
    meta_fd = cfg.meta_resi_fd

n_out = len( labels )

# load model
md = pickle.load( open( 'Md/md100.p', 'rb' ) )

# get wav names to be detected
te_names = ppData.GetWavNamesFromTxt( te_txt )

# do recognize for each test audio
names = os.listdir( fe_fd )
names = sorted( names )
y_pred_list = []

f_ary = []
for na in names:
    if na[0:4] in te_names:
        print na
        gt_file = meta_fd + '/' + na[0:4] + '.ann'
        gt_list = ppData.LoadGtAnn( gt_file )
        
        X = cPickle.load( open( fe_fd+'/'+na, 'rb' ) )
        X = mat_2d_to_3d( X, agg_num, hop )
        y_pred = md.predict( X )
        y_pred_list.append( y_pred )
        output = ppData.OutMatToList( y_pred, thres, id_to_lb )
        
        eva = DCASE2016_EventDetection_SegmentBasedMetrics( labels )
        r = eva.evaluate( gt_list, output ).results()
        f_ary.append( r['overall']['F'] )
    
print 'avg F value: ', np.mean(f_ary)
