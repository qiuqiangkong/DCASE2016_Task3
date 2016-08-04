'''
SUMMARY:  Train model on all development set. 
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 
--------------------------------------
'''
import sys
sys.path.append( '/user/HS229/qk00006/my_code2015.5-/python/Hat' )
import numpy as np
import config as cfg
import prepareData as ppData
from Hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Dropout, Flatten
from Hat.callbacks import SaveModel, Validation
from Hat.optimizers import SGD, Rmsprop, Adam
import prepareData_eva as ppData_eva


### hyper-params
type = 'home'   # can be 'home' or 'resi'
agg_num = 10
hop = 10
n_hid = 500

### train model from all dev data
def train_model():
    # init path
    if type=='home':
        fe_fd = cfg.fe_mel_home_fd
        labels = cfg.labels_home
        lb_to_id = cfg.lb_to_id_home
        meta_fd = cfg.meta_home_fd
    if type=='resi':
        fe_fd = cfg.fe_mel_resi_fd
        labels = cfg.labels_resi
        lb_to_id = cfg.lb_to_id_resi
        meta_fd = cfg.meta_resi_fd
        
    n_out = len( labels )
        
    # load data to list
    tr_X, tr_y = ppData_eva.LoadAllData( fe_fd, meta_fd, lb_to_id, agg_num, hop )
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
    
    ### optimization method
    #optimizer = SGD( lr=0.01, rho=0.9 )
    optimizer = Adam(1e-3)
    
    ### callbacks (optional)
    # save model every n epoch
    save_model = SaveModel( dump_fd='Md_eva', call_freq=10 )
    
    # validate model every n epoch
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metric_types=['categorical_error'], call_freq=1, dump_path='validation.p' )
    
    # callbacks function
    callbacks = [validation, save_model]
    
    ### train model
    md.fit( x=tr_X, y=tr_y, batch_size=20, n_epoch=101, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )
    
### main function
if __name__ == '__main__':
    train_model()