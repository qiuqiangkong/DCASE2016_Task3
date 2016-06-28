'''
SUMMARY:  detect each wav in evaluation set
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: -
--------------------------------------
'''
import sys
sys.path.append( '/homes/qkong/my_code2015.5-/python/Hat' )
sys.path.append( 'evaluation_lib' )
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
type = 'resi'   # can be 'home' or 'resi'
agg_num = 10
hop = 1
thres = 0.99

# init paths
if type=='home':
    fe_fd = cfg.eva_fe_mel_home_fd
    labels = cfg.labels_home
    lb_to_id = cfg.lb_to_id_home
    id_to_lb = cfg.id_to_lb_home
if type=='resi':
    fe_fd = cfg.eva_fe_mel_resi_fd
    labels = cfg.labels_resi
    lb_to_id = cfg.lb_to_id_resi
    id_to_lb = cfg.id_to_lb_resi

n_out = len( labels )

# load model
md = pickle.load( open( 'Md_eva/md100.p', 'rb' ) )

# do recognize for each test audio
names = os.listdir( fe_fd )
names = sorted( names )


fwrite = open('Results_eva/task3_results_' + type + '.txt', 'w')
for na in names:
    X = cPickle.load( open( fe_fd+'/'+na, 'rb' ) )
    X = mat_2d_to_3d( X, agg_num, hop )
    y_pred = md.predict( X )
    outlist = ppData.OutMatToList( y_pred, thres, id_to_lb )
    
    full_na = type + '/audio/' + na[0:4] + '.wav'
    for li in outlist:
        fwrite.write( full_na + '\t' + str(li['event_onset']) + '\t' + str(li['event_offset']) + '\t' + li['event_label'] + '\n' )

fwrite.close()