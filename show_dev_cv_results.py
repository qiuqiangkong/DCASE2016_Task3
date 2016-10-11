'''
SUMMARY:  show results from detected files
AUTHOR:   Qiuqiang Kong
Created:  2016.08.04
Modified: 2016.10.11 modify variable name, fix bug
--------------------------------------
'''
import os
import config as cfg
from subprocess import call
from main_dev_dnn import type
import prepare_dev_data as pp_dev_data
import sed_eval

# ground truth files
if type=='home':
    dev_meta_fd = cfg.dev_meta_home_fd
    gt_files = os.listdir( cfg.dev_meta_home_fd )
    labels = cfg.labels_home
if type=='resi':
    dev_meta_fd = cfg.dev_meta_resi_fd
    gt_files = os.listdir( cfg.dev_meta_resi_fd )
    labels = cfg.labels_resi

# out files
out_files = os.listdir( cfg.dev_results_fd )

# get file list for calcuating results
file_list = []
for na in gt_files:
    gt_file = dev_meta_fd + '/' + na
    out_file = cfg.dev_results_fd + '/' + na[0:4] + '_detect.ann'
    file_list.append( { 'reference_file': gt_file, 'estimated_file': out_file } )

# print results
try:
    pp_dev_data.PrintScore( file_list, labels )
except:
    print 'Error! You should run all 4 folds before using this file!'