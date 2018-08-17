#!/python
""" Download script for google drive shared links 

Thank you @turdus-merula and Andrew Hundt! https://stackoverflow.com/a/39225039/623735
"""
import sys
import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    if 'id=' in id:
        # https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs  # dailymail_stories.tgz
        id = id.split('=')[-1]

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

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

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
