'''
SUMMARY:  prepare data for evaluation data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.27
Modified: -
--------------------------------------
'''
import os
import cPickle
import numpy as np
import config as cfg
import prepareData as ppData
from Hat.preprocessing import mat_2d_to_3d

# load training data and label
def LoadAllData( fe_fd, ann_fd, lb_to_id, agg_num, hop ):
    # anno names
    names = os.listdir( ann_fd )
    names = sorted( names )
    
    # init space
    Xlist, ylist = [], []
    
    # each anno file
    for na in names:
        fr = open( ann_fd+'/'+na, 'r' )
        for line in fr.readlines():
            line_list = line.split('\t')
            
            # parse info
            bgn, fin, lb = float(line_list[0]), float(line_list[1]), line_list[2].split('\n')[0]
            
            # load whole feature
            fe_path = fe_fd + '/' + na[0:4] + '.f'
            X = cPickle.load( open( fe_path, 'rb' ) )
            
            # get sub feature
            ratio = cfg.fs / cfg.win
            X = X[ int(bgn*ratio):int(fin*ratio), : ]
            
            # aggregate feature
            X3d = mat_2d_to_3d( X, agg_num, hop )
            
            Xlist.append( X3d )
            ylist += [ lb_to_id[lb] ] * len(X3d)
        
        fr.close()
        
    return np.concatenate( Xlist, axis=0 ), np.array( ylist )

if __name__ == "__main__":
    ppData.CreateFolder( 'Fe_eva' )
    ppData.CreateFolder( 'Fe_eva/Mel' )
    ppData.CreateFolder( 'Fe_eva/Mel/home' )
    ppData.CreateFolder( 'Fe_eva/Mel/resi' )
    ppData.CreateFolder( 'Results_eva' )
    ppData.CreateFolder( 'Md_eva' )
    
    ppData.GetMel( cfg.eva_wav_home_fd, cfg.eva_fe_mel_home_fd, n_delete=0 )
    ppData.GetMel( cfg.eva_wav_resi_fd, cfg.eva_fe_mel_resi_fd, n_delete=0 )