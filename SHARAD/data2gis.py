import numpy as np
import pandas as pd
import geopandas as gpd
import SHARADEnv

from shapely.geometry import Point, Polygon


def gather(orbitfile, dataprod):
    """Gather data from a list of orbits
    """
    # Gather file names
    df = pd.read_csv(orbitfile, header=None, dtype='object')

    senv = SHARADEnv.SHARADEnv()
    orbitinfo = senv.orbitinfo

    requested = np.array(df[0])
    available = np.array([i for i in orbitinfo.keys()], dtype='object')

    x = np.isin(available, requested)

    orbits = []
    files = []
    for orbit in requested:
        if orbit in available:
            if any('.txt' in s for s in orbitinfo[orbit][0][dataprod+'path']):
                _ = orbitinfo[orbit][0][dataprod+'path']
                for i in _:
                    orbits.append(i)
                    files.append(i)

    # Concatenate data
    out = pd.DataFrame()
    for i in files:
        _ = pd.read_csv(i)
        charar = np.ndarray(len(_)).astype('int')
        charar[:] = orbit
        _['orbit'] = charar
        out = pd.concat([out, _])

    return out


def processor(orbitfile, dataprod, filename='shapefile.shp', proj="E"):
    """Convert a Pandas DataFrame to a GeoDataFrame
    """
    df = gather(orbitfile, dataprod)

    # Conversion
    df['geometry'] = df.apply(lambda x: Point((float(x.lon), float(x.lat))), axis=1)
    df = gpd.GeoDataFrame(df, geometry='geometry')

    # Projection
    if proj == 'E':
        crs = "+proj=longlat +a=3396190 +b=3376200 +no_defs"
    elif proj == 'N':
        crs = "+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs"
    elif proj == 'S':
        crs = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396190 +b=3376200 +units=m +no_defs"
    df.crs = crs

    # Archive
    df.to_file(filename, driver='ESRI Shapefile')


def main():
    parser = argparse.ArgumentParser(description='Convert data files to point Shapefiles for GIS softwares')


if __name__ == "__main__":
    # execute only if run as a script
    main()
