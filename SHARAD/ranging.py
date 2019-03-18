import numpy as np
import pandas as pd
import spiceypy as spice
import matplotlib.pyplot as plt
from misc import coord
from scipy.constants import c
from scipy.ndimage.interpolation import shift
from misc.hdf import hdf
import cmp.pds3lbl as pds3
 
def is_outlier(points, thresh=1):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.nanmedian(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.nanmedian(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def find(name, path='/disk/kea/SDS/orig/supl/xtra-pds/'):
    import os
    for root, dirs, files in os.walk(path, followlinks = True):
        if name in files:
            return os.path.join(root, name)

science_label = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
aux_label = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'
group = 'sharad'
h5file = hdf('cor_mc11.h5', mode='r')
data = h5file.to_dict(group)
h5file.close()
xover = np.load('xover/cor_mc11e_xover.npy')
res = []
test = []

count = 0
for x in xover:
    #===============================================
    #  READ INPUT
    #===============================================
    track1 = data[x[0]]
    track2 = data[x[1]]
    aux1  = pds3.read_science(find(x[0]+'.dat'), aux_label, science=False)
    aux2  = pds3.read_science(find(x[1]+'.dat'), aux_label, science=False)
    idx1   = x[2]
    idx2   = x[3]
    param11 = aux1['SC_PITCH_ANGLE'][idx1]
    param12 = aux2['SC_PITCH_ANGLE'][idx2]
    param21 = aux1['SC_YAW_ANGLE'][idx1]
    param22 = aux2['SC_YAW_ANGLE'][idx2]
    r1 = track1['spot_radius'].as_matrix()
    r2 = track2['spot_radius'].as_matrix()
    truth1 = np.load('cor_truth/'+x[0]+'.npy')[:,3]
    truth2 = np.load('cor_truth/'+x[1]+'.npy')[:,3]
    idx_start1 = max(0,idx1-500)
    idx_end1 = min(len(r1),idx1+500)
    idx_start2 = max(0,idx2-500)
    idx_end2 = min(len(r2),idx2+500)
    alt1 = r1[idx_start1:idx_end1]
    alt2 = r2[idx_start2:idx_end2]
    diff1 = alt1[0] - truth1[idx_start1]
    diff2 = alt2[0] - truth2[idx_start2]
    alt1[np.where(is_outlier(r1[idx_start1:idx_end1]) == True)]=np.nan
    alt2[np.where(is_outlier(r2[idx_start2:idx_end2]) == True)]=np.nan
    alt1 = alt1-3389E+3
    alt2 = alt2-3389E+3
    print(np.nanmean(alt1),np.nanmean(alt2),np.nanmean(alt1)-np.nanmean(alt2))
    #if abs(np.nanmean(alt1)-np.nanmean(alt2))>20000:
    #    plt.scatter(np.arange(len(alt1)),alt1,s=0.1)
    #    plt.scatter(np.arange(len(alt2)),alt2,s=0.1)
    #    plt.show()
    res.append(np.nanmean(alt1)-np.nanmean(alt2))
    test.append([param11,param21,diff1])
    test.append([param12,param22,diff2])
    count+=1
    if count>500: break
res = np.array(res)
test = np.array(test)
plt.scatter(test[:,0],test[:,2],s=0.3)
plt.show()
plt.scatter(test[:,1],test[:,2],s=0.3)
plt.show()
plt.scatter(np.arange(len(res)),res,s=0.3)
plt.show()

res[np.where(is_outlier(res) == True)]=np.nan
print(np.nanmean(res),np.nanstd(res))

