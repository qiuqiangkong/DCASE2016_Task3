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
          2016.10.11 Modify variable name
--------------------------------------
'''
import numpy as np
import config as cfg
import prepare_dev_data as pp_dev_data
from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, mat_2d_to_3d
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Flatten
from hat.callbacks import SaveModel, Validation
from hat.optimizers import Adam
import hat.backend as K


### hyper-params
fold = 1        # can be 1,2,3 or 4
type = 'resi'   # can be 'home' or 'resi'
agg_num = 11
hop = 5
n_hid = 500


### train model from cross validation data
def train_cv_model():
    # init path
    if type=='home':
        fe_fd = cfg.dev_fe_mel_home_fd
        labels = cfg.labels_home
        lb_to_id = cfg.lb_to_id_home
        tr_txt = cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_train.txt'
        te_txt = cfg.dev_evaluation_fd + '/home_fold' + str(fold) + '_evaluate.txt'
    if type=='resi':
        fe_fd = cfg.dev_fe_mel_resi_fd
        labels = cfg.labels_resi
        lb_to_id = cfg.lb_to_id_resi
        tr_txt = cfg.dev_evaluation_fd + '/residential_area_fold' + str(fold) + '_train.txt'
        te_txt = cfg.dev_evaluation_fd + '/residential_area_fold' + str(fold) + '_evaluate.txt'
        
    n_out = len( labels )
    
    # load data to list
    tr_X, tr_y = pp_dev_data.LoadAllData( fe_fd, tr_txt, lb_to_id, agg_num, hop )
    tr_y = sparse_to_categorical( tr_y, n_out )
    
    print tr_X.shape
    print tr_y.shape
    n_freq = tr_X.shape[2]
    
    # build model
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
    
    # callbacks (optional)
    # save model every n epoch
    pp_dev_data.CreateFolder( cfg.dev_md_fd )
    save_model = SaveModel( dump_fd=cfg.dev_md_fd, call_freq=5 )
    
    # validate model every n epoch
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metrics=['binary_crossentropy'], call_freq=1, dump_path=None )
    
    # callbacks function
    callbacks = [validation, save_model]
    
    # train model
    md.fit( x=tr_X, y=tr_y, batch_size=20, n_epochs=100, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )


### main function
if __name__ == '__main__':
    train_cv_model()