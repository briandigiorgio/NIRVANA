import os
import warnings
import requests
import netrc
from astropy.io import fits

try:
    from tqdm import tqdm
except:
    tqdm = None

try:
    NETRC = netrc.netrc()
except Exception as e:
    raise FileNotFoundError('Could not load ~/.netrc file.') from e

def download_file(url, user, password, outfile, clobber=True):
    """
    Thanks to 
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    """
    #Beware of how this is joined!
    if os.path.isfile(outfile):
        if clobber:
            warnings.warn('Overwriting existing file: {0}'.format(outfile))
            os.remove(outfile)
        else:
            warnings.warn('Using already existing file. To overwrite, set overwrite=True.')
            return

    print('Downloading: {0}'.format(url))
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, auth=(user, password))
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(outfile, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        raise ValueError('Downloaded file may be corrupted.')

def download_plateifu(plate, ifu, outdir, dr='MPL-11', daptype='HYB10-MILESHC-MASTARHC2', basedir='https://data.sdss.org/sas/mangawork/manga/spectro', clobber=True):
    user, acc, password = NETRC.authenticators('data.sdss.org')
    outpath = f'{outdir}/{plate}/{ifu}'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    fname = f'manga-{plate}-{ifu}-MAPS-{daptype}.fits.gz'
    url = f'{basedir}/analysis/{dr}/{daptype}/{plate}/{ifu}/{fname}'
    mapsfile = f'{outpath}/{fname}'
    download_file(url, user, password, mapsfile, clobber)
    try: fits.open(mapsfile)
    except Exception as e: 
        raise ValueError('Downloaded MAPS file may be corrupted.') from e


    fname = f'manga-{plate}-{ifu}-LOGCUBE.fits.gz'
    url = f'{basedir}/redux/{dr}/{plate}/stack/{fname}'
    cubefile = f'{outpath}/{fname}'
    download_file(url, user, password, cubefile, clobber)
    try: fits.open(cubefile)
    except Exception as e: 
        raise ValueError('Downloaded LOGCUBE file may be corrupted.') from e

    url = f'{basedir}/redux/{dr}/{plate}/images/{ifu}.png'
    imfile = f'{outpath}/{ifu}.png'
    download_file(url, user, password, imfile, clobber)

    return mapsfile, cubefile, imfile
