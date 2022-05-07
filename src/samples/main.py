from nis import match
import os
import argparse
import glob
import re

here = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, required=False, default=False)
    args = parser.parse_args()


    return args

def fetch_files(name):
    """

    - patient_name: According to the styleguide, the csv file that contains older measurement
                    is named after the patients name. Create the file is it does not yet exist.
    
    Returns tuple that contains the csv path that corresponds to the given patient_name
    and a list of video paths. 
    """

    csv_folder = os.path.join(here, 'data', 'csv')
    video_folder = os.path.join(here, 'data', 'videos')

    # Get path to csv file
    csv_path = os.path.join(csv_folder, name + '.csv')
    if not os.path.exists(csv_path):
        # Give warning if csv file does not exist.
        # The video_analyzer will create it later anyway.
        print('CSV file for {} does not exist.\nNew file will be created @ {}'.format(name, csv_path))

    # Create list with files in the video folder.
    # TODO: add checks for video formats.
    video_files = []
    for filename in glob.glob(os.path.join(video_folder, '*')):
        if len(re.findall(r'\([1-9]\)', filename)) != 0: video_files.append(filename)
    
    # Numerically sort filenames based on the number in the filename.
    # Splits on the space between name and (X). The second index
    # takes the X part out of (X).
    #
    # Example filename:
    # '/home/robbedec/Desktop/masterproef_debug/data/videos/test (1).avi'
    video_files.sort(key=lambda x: int(x.split()[1][1]))

    return csv_path, video_files


def main():
    patient_name = 'robbe' if debug else input('Geef de naam van de patient: ')
    fetch_files(name=patient_name)

if __name__ == '__main__':
    args = get_args()

    # Enable debug mode if indicated
    global debug
    debug = args.debug

    if debug:
        here = '/home/robbedec/Desktop/masterproef_debug'
    
    main()