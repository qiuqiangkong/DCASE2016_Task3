'''
SUMMARY:  Train model on all development set. 
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 2016.10.11 modify variable name
--------------------------------------
'''
import sys
import numpy as np
import config as cfg
from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Flatten
from hat.callbacks import SaveModel, Validation
from hat.optimizers import SGD, Rmsprop, Adam
import prepare_dev_data as pp_dev_data
import prepare_eva_data as pp_eva_data


### hyper-params
type = 'home'   # can be 'home' or 'resi'
agg_num = 11
hop = 5
n_hid = 500

### train model from all dev data
def train_model():
    # init path
    if type=='home':
        fe_fd = cfg.dev_fe_mel_home_fd
        labels = cfg.labels_home
        lb_to_id = cfg.lb_to_id_home
        meta_fd = cfg.dev_meta_home_fd
    if type=='resi':
        fe_fd = cfg.dev_fe_mel_resi_fd
        labels = cfg.labels_resi
        lb_to_id = cfg.lb_to_id_resi
        meta_fd = cfg.dev_meta_resi_fd
        
    n_out = len( labels )
        
    # load data to list
    tr_X, tr_y = pp_eva_data.LoadAllData( fe_fd, meta_fd, lb_to_id, agg_num, hop )
    tr_y = sparse_to_categorical( tr_y, n_out )
    print tr_X.shape, tr_y.shape
    
    # build model
    n_freq = tr_X.shape[2]
    seq = Sequential()
    seq.add( InputLayer( (agg_num, n_freq) ) )
    seq.add( Flatten() )
    seq.add( Dense( n_hid, act='relu' ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_hid, act='relu' ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_hid, act='relu' ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( n_out, 'sigmoid' ) )
    md = seq.combine()
    
    # print summary info of model
    md.summary()
    
    # optimization method
    optimizer = Adam(1e-3)
    
    # validation
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metrics=['categorical_error'], call_freq=1, dump_path=None )
    
    # save model
    pp_dev_data.CreateFolder( cfg.eva_md_fd )
    save_model = SaveModel( dump_fd=cfg.eva_md_fd, call_freq=5 )
    
    # callbacks
    callbacks = [validation, save_model]
    
    # train model
    md.fit( x=tr_X, y=tr_y, batch_size=20, n_epochs=100, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )
    
### main function
if __name__ == '__main__':
    train_model()