import numpy as np
import matplotlib.pyplot as plt
import os 
from mpl_toolkits.basemap import Basemap, cm
m = Basemap(projection='npstere',boundinglat=60,lon_0=270,resolution='l')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='white')

res = 1
j=0
length = len(os.listdir('alt_N/'))
grid = np.zeros((int(360*res),int(30*res)))
nb = np.zeros((int(360*res),int(30*res)))
for file in os.listdir('alt_N/'):
    track = np.load('alt_N/'+file)
    lat = track[:,0]
    lon = track[:,1]
    data = track[:,2]
    for i in range(len(track)):
        grid[int(lon[i]*res),int((lat[i]-60)*res)] = data[i]
        nb[int(lon[i]*res),int((lat[i]-60)*res)] += 1
    j+=1
    print(j,'/',length)
avg_map = grid/nb
nx = np.arange(0,360,1.0/res)
ny = np.arange(60,90,1.0/res)
lon, lat = np.meshgrid(nx, ny)
m.pcolormesh(lon, lat, avg_map.transpose(), latlon=True, vmin=-1,vmax=1)
ax = plt.gca()
plt.colorbar() 
plt.show()

