import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
from shapely.geometry import Point

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
data = Path(r'/mnt/data/banana_data')
print('Loading data...')
grid = gpd.read_file(data.joinpath(r'grid.geojson'))
tin = gpd.read_file(data.joinpath(r'tin.geojson'))
orthofoto = rio.open(data.joinpath('orthofoto.vrt'))

dim = 1024
cell_size = 0.1
crs = orthofoto.crs


# -----------------------------------------------------------------------------
# ZZZZ
# -----------------------------------------------------------------------------
def calculate_z(id, left, top, triangles, meta):
    t_start = time.time()
    image = np.zeros((1, dim, dim))
    dest = data.joinpath("dtm/{}.tif".format(id))
    try:
        for i in range(0, dim, 8):
            for j in range(0, dim, 8):
                x = left + (0.5 + j) * cell_size
                y = top - (0.5 + i) * cell_size
                intersects = triangles.intersects(Point(x, y))
                if intersects.sum() == 0:
                    continue
                tid = triangles[intersects].index.tolist()[0]
                coords = np.array(
                    triangles.loc[tid, 'geometry'].exterior.coords)
                A, B, C, X, Y, Z = 0, 1, 2, 0, 1, 2
                image[0, i, j] = coords[A, Z] + (
                    (((coords[B, X] - coords[A, X]) * (coords[C, Z] - coords[A, Z]))
                     - ((coords[C, X] - coords[A, X]) * (coords[B, Z] - coords[A, Z])))
                    / (((coords[B, X] - coords[A, X]) * (coords[C, Y] - coords[A, Y]))
                     - ((coords[C, X] - coords[A, X]) * (coords[B, Y] - coords[A, Y])))
                ) * (y - coords[A, Y]) - (
                    (((coords[B, Y] - coords[A, Y]) * (coords[C, Z] - coords[A, Z]))
                     - ((coords[C, Y] - coords[A, Y]) * (coords[B, Z] - coords[A, Z])))
                    / (((coords[B, X] - coords[A, X]) * (coords[C, Y] - coords[A, Y]))
                     - ((coords[C, X] - coords[A, X]) * (coords[B, Y] - coords[A, Y])))
                ) * (x - coords[A, X])
    except Exception as e:
        print('Unable to process', id, e)
    else:
        with rasterio.open(dest, "w", **meta) as f:
            f.write(image.astype('float32'))
        print('Extracted tile {} in {} minutes'.format(
            id, (time.time() - t_start) / 60))
        return dest


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    pool = Pool(10)
    print('Start processing...')
    count = 0
    for index, row in grid.iterrows():
        geom = grid.loc[index, 'geometry']
        id = grid.loc[index, 'id']
        left = grid.loc[index, 'left']
        top = grid.loc[index, 'top']
        if data.joinpath('dtm/{}.tif'.format(id)).exists():
            continue
        triangles = tin[tin.intersects(geom.buffer(1))].copy()
        _, transform = rio.mask.mask(orthofoto, [geom], crop=True)
        meta = {"driver": "GTiff", 'dtype': 'float32', 'nodata': None,
                "height": dim, "width": dim, 'count': 1, 'crs': crs,
                "transform": transform}
        pool.apply_async(calculate_z, (id, left, top, triangles, meta))
        #calculate_z(id, left, top, triangles, meta)
