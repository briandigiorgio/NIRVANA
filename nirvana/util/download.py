import os
import warnings
import requests
import tqdm


def download_file(url, outfile, overwrite=True, auth=None):
    """
    Download a file.

    Thanks to 
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701

    Args:
        url (:obj:`str`):
            Full URL to the file to download.
        outfile (:obj:`str`):
            Full path for the downloaded file.
        overwrite (:obj:`bool`, optional):
            If the file exists, overwrite it.
        auth (:obj:`tuple`, optional):
            If the download requires authentication, this is, e.g., a tuple
            with the user name and password. This is passed directly to
            `requests.get`_, meaning any accepted format in that function is
            also accepted here.

    Raises:
        ValueError:
            Raised if the file may have been corrupted on transfer.
    """
    #Beware of how this is joined!
    if os.path.isfile(outfile):
        if overwrite:
            warnings.warn('Overwriting existing file: {0}'.format(outfile))
            os.remove(outfile)
        else:
            warnings.warn(f'{outfile} exists. To overwrite, set overwrite=True.')
            return

    print('Downloading: {0}'.format(url))
    print('To: {0}'.format(outfile))
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, auth=auth)
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
        raise ValueError(f'Downloaded file ({outfile}) may be corrupted.')


