'''
SUMMARY:  detect each wav in evaluation set
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 2016.08.04 add to detect() function
          2016.10.11 modify variable name
--------------------------------------
'''
import sys
import numpy as np
import config as cfg
import prepare_dev_data as pp_dev_data
from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Flatten
from hat.callbacks import SaveModel, Validation
from hat.optimizers import SGD, Rmsprop
from hat import serializations
import hat.backend as K
import os
import cPickle
import pickle
import matplotlib.pyplot as plt
from main_eva_dnn import type, agg_num, hop

### hyper-params
thres = 0.99
md_path = cfg.eva_md_fd + '/md10.p'

def detect():
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
    md = serializations.load( md_path )
    
    # do recognize for each test audio
    names = os.listdir( fe_fd )
    names = sorted( names )
    pp_dev_data.CreateFolder( cfg.eva_results_fd )
    pp_dev_data.CreateFolder( cfg.eva_results_fd+'/'+type )
    
    # detect and write out for all audios
    for na in names:
        X = cPickle.load( open( fe_fd+'/'+na, 'rb' ) )
        X = mat_2d_to_3d( X, agg_num, hop )
        y_pred = md.predict( X )
        outlist = pp_dev_data.OutMatToList( y_pred, thres, id_to_lb )
        
        full_na = type + '/audio/' + na[0:4] + '.wav'
        out_txt_path = cfg.eva_results_fd+'/'+type+'/'+na[0:4]+'_detect.ann'
        f = open( out_txt_path, 'w')
        for li in outlist:
            f.write( full_na + '\t' + str(li['event_onset']) + '\t' + str(li['event_offset']) + '\t' + li['event_label'] + '\n' )
        print 'Write out detection result to', out_txt_path, 'successfully!'
    
    f.close()


### main function
if __name__ == '__main__':
    detect()