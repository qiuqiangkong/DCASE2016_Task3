'''
SUMMARY:  show results from detected files
AUTHOR:   Qiuqiang Kong
Created:  2016.08.04
Modified: -
--------------------------------------
'''
import os
import config as cfg
from subprocess import call
from main_dnn import type
import prepareData as ppData
import sed_eval

# ground truth files
if type=='home':
    gt_files = os.listdir( cfg.meta_home_fd )
    labels = cfg.labels_home
if type=='resi':
    gt_files = os.listdir( cfg.meta_resi_fd )
    labels = cfg.labels_resi

# out files
out_files = os.listdir( cfg.results_fd )

# get file list for calcuating results
file_list = []
for na in gt_files:
    gt_file = cfg.meta_home_fd + '/' + na
    out_file = cfg.results_fd + '/' + na[0:4] + '_detect.ann'
    file_list.append( { 'reference_file': gt_file, 'estimated_file': out_file } )

# print results
try:
    ppData.PrintScore( file_list, labels )
except:
    print 'Error! You should run all 4 folds before using this file!'