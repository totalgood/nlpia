#!/python
""" Download script for google drive shared links 

Thank you @turdus-merula and Andrew Hundt! 
https://stackoverflow.com/a/39225039/623735
"""
from nlpia.web import download_file_from_google_drive


def main():
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        file_id = sys.argv[1]  # TAKE ID FROM SHAREABLE LINK
        destination = sys.argv[2]  # DESTINATION FILE ON YOUR DISK
        download_file_from_google_drive(file_id, destination)


def main(driveid=None, filename=None):
    if driveid is None:
        if len(sys.argv) < 2:
            print("Usage: python google_drive.py drive_file_id destination_file_path")
            return
        else:
            driveid = sys.argv[1]  # TAKE ID FROM SHAREABLE LINK
    if filename is None:
        if len(sys.argv) > 2:
            filename = sys.argv[2]  # DESTINATION FILE ON YOUR DISK
    download_file_from_google_drive(driveid, filename)


if __name__ == "__main__":
    main()
