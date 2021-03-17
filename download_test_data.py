
import os
import warnings
import tqdm
import requests
import netrc

from nirvana.tests.util import remote_data_file, remote_drp_test_files, remote_drp_test_images
from nirvana.tests.util import remote_dap_test_files
from nirvana.tests.util import drp_test_version, dap_test_version, dap_test_daptype

try:
    NETRC = netrc.netrc()
except Exception as e:
    raise FileNotFoundError('Could not load ~/.netrc file.') from e

HOST = 'data.sdss.org'
if HOST not in NETRC.hosts:
    raise ValueError('Host data.sdss.org is not defined in your ~/.netrc file.')


def download_file(remote_root, usr, passwd, local_root, file, overwrite=True):
    """
    Thanks to 
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    """
    #Beware of how this is joined!
    url = '{0}{1}'.format(remote_root, file)
    ofile = os.path.join(local_root, file)

    if os.path.isfile(ofile):
        if overwrite:
            warnings.warn('Overwriting existing file: {0}'.format(ofile))
            os.remove(ofile)
        else:
            raise FileExistsError('File already exists.  To overwrite, set overwrite=True.')

    print('Downloading: {0}'.format(url))
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, auth=(usr, passwd))
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(ofile, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        raise ValueError('Downloaded file may be corrupted.')


def main():

    from IPython import embed

    usr, acc, passwd = NETRC.authenticators(HOST)

    local_root = remote_data_file()
    if not os.path.isdir(local_root):
        os.makedirs(local_root)

    # DRP files
    drp_files = remote_drp_test_files()
    drp_images = remote_drp_test_images()
    plates = [f.split('-')[1] for f in drp_files]
    for plate, fcube, fimg in zip(plates, drp_files, drp_images):
        if os.path.isfile(os.path.join(local_root, fcube)):
            warnings.warn('{0} exists.  Skipping...'.format(fcube))
        else:
            url_root = 'https://{0}/sas/mangawork/manga/spectro/redux/{1}/{2}/stack/'.format(
                            HOST, drp_test_version, plate)
            download_file(url_root, usr, passwd, local_root, fcube)
        if os.path.isfile(os.path.join(local_root, fimg)):
            warnings.warn('{0} exists.  Skipping...'.format(fimg))
        else:
            url_root = 'https://{0}/sas/mangawork/manga/spectro/redux/{1}/{2}/images/'.format(
                            HOST, drp_test_version, plate)
            download_file(url_root, usr, passwd, local_root, fimg)

    # DAP files
    dap_files = remote_dap_test_files(daptype=dap_test_daptype)
    plates = [f.split('-')[1] for f in dap_files]
    ifus = [f.split('-')[2] for f in dap_files]
    for plate, ifu, f in zip(plates, ifus, dap_files):
        if os.path.isfile(os.path.join(local_root, f)):
            warnings.warn('{0} exists.  Skipping...'.format(f))
            continue
        url_root = 'https://{0}/sas/mangawork/manga/spectro/analysis/{1}/{2}/{3}/{4}/{5}/'.format(
                        HOST, drp_test_version, dap_test_version, dap_test_daptype, plate, ifu)
        download_file(url_root, usr, passwd, local_root, f)

    # DRPall file
    f = 'drpall-{0}.fits'.format(drp_test_version)
    url_root = 'https://{0}/sas/mangawork/manga/spectro/redux/{1}/'.format(HOST, drp_test_version)
    if os.path.isfile(os.path.join(local_root, f)):
        warnings.warn('{0} exists.  Skipping...'.format(f))
    else:    
        download_file(url_root, usr, passwd, local_root, f)

if __name__ == '__main__':
    main()


