'''
SUMMARY:  Event detection using threshold, return list of <bgn, fin>
AUTHOR:   Qiuqiang Kong
Created:  2016.06.15
Modified: -
--------------------------------------
'''
import numpy as np

'''
find pairs of <bgn, fin> from input array
'''
def activity_detection( x, thres, n_smooth=1 ):
    locts = np.where( x>thres )[0]
    locts = smooth( locts, n_smooth )
    lists = find_bgn_fin_pairs( locts )
    return lists

'''
Smooth the loctation array
Params:
  locts      location array
  n_smooth   number of points to smooth
Return:
  locts      smoothed location array
Eg.
  input: np.array([3,4,7,8])
  return: np.array([3,4,5,6,7,8])
'''
def smooth( locts, n_smooth ):
    if len(locts)==0:
        return locts
    else:
        smooth_locts = [ locts[0] ]
        for i1 in xrange(1,len(locts)):
            if locts[i1]-locts[i1-1] <= n_smooth:                
                for i2 in xrange( locts[i1-1]+1, locts[i1] ):
                    smooth_locts.append( i2 )
            smooth_locts.append( locts[i1] )
        return smooth_locts
        
'''
Find pairs of <bgn, fin> from loctation array
'''
def find_bgn_fin_pairs( locts ):
    if len(locts)==0:
        return []
    else:
        bgns = [ locts[0] ]
        fins = []
        for i1 in xrange(1,len(locts)):
            if locts[i1]-locts[i1-1]>1:
                fins.append( locts[i1-1] )
                bgns.append( locts[i1] )
        fins.append( locts[-1] )
            
    assert len(bgns)==len(fins)
    lists = []
    for i1 in xrange( len(bgns) ):
        lists.append( { 'bgn':bgns[i1], 'fin':fins[i1] } )
    return lists
    