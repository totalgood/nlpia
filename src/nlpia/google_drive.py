#!/python
""" Download script for google drive shared links 

Thank you @turdus-merula and Andrew Hundt! 
https://stackoverflow.com/a/39225039/623735
"""
import sys
import requests
from tqdm import tqdm
from nlpia.loaders import get_url_title


def get_url_filename(url=None, driveid=None):
    url = url or 'https://drive.google.com/open?id={}'.format(driveid)
    if url.startswith('https://drive.google.com'):
        filename = get_url_title(url)
        if filename.endswith('Google Drive'):
            filename = filename[:-len('Google Drive')].rstrip().rstrip('-:').rstrip()
        return filename
    


def download_file_from_google_drive(driveid, destination=None):
    if '&id=' in driveid:
        # https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs  # dailymail_stories.tgz
        driveid = driveid.split('&id=')[-1]
    if '?id=' in driveid:
        # 'https://drive.google.com/open?id=14mELuzm0OvXnwjb0mzAiG-Ake9_NP_LQ'  # SSD pretrainined keras model
        driveid = driveid.split('?id=')[-1]

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': driveid}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': driveid, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    destination = destination or get_url_filename(driveid=driveid)

    save_response_content(response, destination)    


def main():
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        file_id = sys.argv[1]  # TAKE ID FROM SHAREABLE LINK
        destination = sys.argv[2]  # DESTINATION FILE ON YOUR DISK
        download_file_from_google_drive(file_id, destination)


if __name__ == "__main__":
    main()
