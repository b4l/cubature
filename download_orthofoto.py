from pathlib import Path

import requests

data = Path(r'/mnt/data/banana_data')
tilenames = data.joinpath(r'orthofoto_tiles.csv').open().readlines()

format = 'jpeg'  # 'tif' or 'jpeg'
version = 'spring'  # 'spring' or 'summer'


base_url = r'https://maps.zh.ch/download/orthofoto/'
versions = {'summer': 'sommer/2014/rgb/',
            'spring': 'fruehjahr/2015/rgb/'}

for tilename in tilenames:
    if '.tif' not in tilename:
        continue
    tilename = tilename.strip()
    dst = data.joinpath('orthofoto_' + version, tilename)

    url = base_url + versions[version] + format + '/' + tilename
    r = requests.get(url)
    r.raise_for_status()

    Path.mkdir(dst.parent, parents=True, exist_ok=True)
    dst.write_bytes(r.content)
