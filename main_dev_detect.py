'''
SUMMARY:  detect each wav from cross validation
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 2016.08.04
          2016.10.11 modify variable name
--------------------------------------
'''
import numpy as np
import config as cfg
import prepare_dev_data as pp_dev_data
from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Flatten
from hat.callbacks import SaveModel, Validation
from hat import serializations
import hat.backend as K
import os
import cPickle
import pickle
import matplotlib.pyplot as plt
from main_dev_dnn import fold, type, agg_num, hop


### hyper params
thres = 0.1    # if probability pass this value, then label as an event
md_path = cfg.dev_md_fd + '/md10.p'


### detect cross validation test files
def detect_cv():
    # init paths
    if type=='home':
        fe_fd = cfg.dev_fe_mel_home_fd
        labels = cfg.labels_home
        lb_to_id = cfg.lb_to_id_home
        id_to_lb = cfg.id_to_lb_home
        tr_txt = cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_train.txt'
        te_txt = cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_evaluate.txt'
        meta_fd = cfg.dev_meta_home_fd
    if type=='resi':
        fe_fd = cfg.dev_fe_mel_resi_fd
        labels = cfg.labels_resi
        lb_to_id = cfg.lb_to_id_resi
        id_to_lb = cfg.id_to_lb_resi
        tr_txt = cfg.dev_evaluation_fd + '/residential_area_fold' + str(fold) + '_train.txt'
        te_txt = cfg.dev_evaluation_fd + '/residential_area_fold' + str(fold) + '_evaluate.txt'
        meta_fd = cfg.dev_meta_resi_fd
    
    n_out = len( labels )
    
    # load model
    md = serializations.load( md_path )
    
    # get wav names to be detected
    te_names = pp_dev_data.GetWavNamesFromTxt( te_txt )
    
    # do recognize for each test audio
    names = os.listdir( fe_fd )
    names = sorted( names )
    y_pred_list = []

    # detect and write out to txt
    pp_dev_data.CreateFolder( cfg.dev_results_fd )
    file_list = []
    for na in names:
        if na[0:4] in te_names:
            print na
            gt_file = meta_fd + '/' + na[0:4] + '.ann'
            out_file = cfg.dev_results_fd + '/'+na[0:4]+'_detect.ann'
            
            X = cPickle.load( open( fe_fd+'/'+na, 'rb' ) )
            X = mat_2d_to_3d( X, agg_num, hop )
            y_pred = md.predict( X )
            
            y_pred_list.append( y_pred )
            out_list = pp_dev_data.OutMatToList( y_pred, thres, id_to_lb )
            pp_dev_data.PrintListToTxt( out_list, out_file )
            
            file_list.append( { 'reference_file': gt_file, 'estimated_file': out_file } )
            
    # print results for this fold
    pp_dev_data.PrintScore( file_list, labels )
    


### main function
if __name__ == '__main__':
    detect_cv()