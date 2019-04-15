import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cmp import pds3lbl as pds3
from scipy.constants import c

xover = np.load('mc11_xover.npy')
lookup = np.genfromtxt('lookup.txt',dtype='str')
label_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/science_ancillary.fmt'
aux_path = '/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0004/label/auxiliary.fmt'

diff = []
sza = []
for x in xover:
    try:
        gob1 = int(x[0].replace('orbit',''))
        gob2 = int(x[1].replace('orbit',''))
        idx1 = x[2]
        idx2 = x[3]
        # track 1
        path1 = lookup[gob1]
        #path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        #data_file = path_file.split('/')[-1]
        #path_file = path_file.replace(data_file,'')
        #new_path = 'alt2/'+data_file.replace('.dat','.npy')
        #tecu_file = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'+path_file+'ion/'+data_file.replace('_a.dat','_s_TECU.txt')
        #tec1 = np.loadtxt(tecu_file)*1E+15
        #track1 = np.load(new_path)
        aux1 = pds3.read_science(path1.replace('_s.dat','_a.dat'), aux_path, science=False, bc=False)
        sza1 = aux1['SOLAR_ZENITH_ANGLE'][idx1] 
        #delay1 = c*tec1*1.69E-6/(2*np.pi*(20E+6)**2)
        #print(sza1,tec1[0],tec1[0]*1.69E-6/(20E+6)**2)        

        # track 2
        path2 = lookup[gob2]
        #data2 = pds3.read_science(path, label_path, science=True, bc=False)
        aux2 = pds3.read_science(path2.replace('_s.dat','_a.dat'), aux_path, science=False, bc=False)
        sza2=aux2['SOLAR_ZENITH_ANGLE'][idx2]
        #path_file = path.replace('/disk/kea/SDS/orig/supl/xtra-pds/SHARAD/EDR/','')
        #data_file = path_file.split('/')[-1]
        #path_file = path_file.replace(data_file,'')
        #new_path = 'alt2/'+data_file.replace('.dat','.npy')
        #tecu_file = '/disk/kea/SDS/targ/xtra/SHARAD/cmp/'+path_file+'ion/'+data_file.replace('_a.dat','_s_TECU.txt')
        #tec2 = np.loadtxt(tecu_file)*1E+15
        #delay2 = c*tec2*1.69E-6/(2*np.pi*(20E+6)**2)
        #print(sza2,tec2[0],tec2[0]*1.69E-6/(20E+6)**2)        
        #track2 = np.load(new_path)
        # area around x-over plate
        #i1 = max(0,idx1-100)
        #i2 = min(len(track1),idx1+100)
        #j1 = max(0,idx2-100)
        #j2 = min(len(track2),idx2+100)
        #plt.plot(track1[i1:i2,1],track1[i1:i2,0])
        #plt.plot(track2[j1:j2,1],track2[j1:j2,0])
        #plt.show()
        #plt.plot(track1[i1:i2,2])
        #plt.plot(track2[j1:j2,2])
        #plt.show()

        #alt1=track1[idx1,2]+delay1[idx1,0]/2000
        #alt2=track2[idx2,2]+delay2[idx2,0]/2000
        #sza.append(sza1-sza2)
        if sza1>100 and sza2>100: 
            print(sza1,sza2, x)#,alt1,alt2,delay1[idx1,0],delay2[idx2,0])
            print(path1.replace('_s.dat','_a.dat'))
            print(path2.replace('_s.dat','_a.dat'))       
            print()
        #diff.append((alt1-alt2)*1000)
    except Exception as e:
        print(e)
        continue 
plt.scatter(np.arange(len(diff)),diff,s=0.5)
plt.show()
