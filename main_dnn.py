'''
SUMMARY:  DCASE Challenge Task 2, real life aed. 
          Using dnn for training. 
          F value of home on each fold: 28.0%, 28.8%, 22.3%, 37.5%, avg; 29.2%
          F value of resi on each fold: 62.4%, 34.5%, 43.7%, 47.5%; avg; 47.0%
          3 s/epoch on Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.06.10
Modified: 2016.06.13
          2016.06.26
--------------------------------------
'''
import sys
sys.path.append( '/homes/qkong/my_code2015.5-/python/Hat' )
import numpy as np
import config as cfg
import prepareData as ppData
from Hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Dropout, Flatten
from Hat.callbacks import SaveModel, Validation
from Hat.optimizers import SGD, Rmsprop
import Hat.backend as K

# hyper-params
fold = 1        # can be 1,2,3 or 4
type = 'home'   # can be 'home' or 'resi'
agg_num = 10
hop = 5
n_hid = 500

# init path
if type=='home':
    fe_fd = cfg.fe_mel_home_fd
    labels = cfg.labels_home
    lb_to_id = cfg.lb_to_id_home
    tr_txt = cfg.evaluation_fd + '/home_fold' + str(fold) + '_train.txt'
    te_txt = cfg.evaluation_fd + '/home_fold' + str(fold) + '_evaluate.txt'
if type=='resi':
    fe_fd = cfg.fe_mel_resi_fd
    labels = cfg.labels_resi
    lb_to_id = cfg.lb_to_id_resi
    tr_txt = cfg.evaluation_fd + '/residential_area_fold' + str(fold) + '_train.txt'
    te_txt = cfg.evaluation_fd + '/residential_area_fold' + str(fold) + '_evaluate.txt'
    
n_out = len( labels )

# load data to list
'''
Xlist, ylist, bg_list = ppData.LoadListData( fe_fd, tr_txt, lb_to_id )
tr_X, tr_y = ppData.Aggregate( Xlist, ylist, bg_list, agg_num, hop, lb_to_id )
tr_y = sparse_to_categorical( tr_y, n_out )
'''
tr_X, tr_y = ppData.LoadAllData( fe_fd, tr_txt, lb_to_id, agg_num, hop )
tr_y = sparse_to_categorical( tr_y, n_out )

print tr_X.shape
print tr_y.shape
n_freq = tr_X.shape[2]

# build model
md = Sequential()
md.add( InputLayer( (agg_num, n_freq) ) )
md.add( Flatten() )
md.add( Dense( n_hid, act='relu' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act='relu' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act='relu' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_out, 'sigmoid' ) )

# print summary info of model
md.summary()
md.plot_connection()

### optimization method
#optimizer = SGD( lr=0.01, rho=0.9 )
optimizer = Rmsprop(1e-4)

### callbacks (optional)
# save model every n epoch
save_model = SaveModel( dump_fd='Md', call_freq=10 )

# validate model every n epoch
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metric_types=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=20, n_epoch=101, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )