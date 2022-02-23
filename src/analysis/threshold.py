from matplotlib.pyplot import draw
import pandas as pd
import os
import glob
import cv2

from src.analysis.StaticAnalyzer import StaticAnalyzer
from src.utils.util import resize_with_aspectratio
from src.analysis.measurements import Measurements

DATA_PATH = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set'
DATA_PATH_NORMAL = os.path.join(DATA_PATH, 'Normals')
DATA_PATH_FLACCID = os.path.join(DATA_PATH, 'Flaccid')

CSV_MEASUREMENTS_PATH = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/meei_measurements.csv'
CSV_PROCESSED_PATH = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/meei_measurements_processed.csv'

analyzer = StaticAnalyzer()

def analyze_img(img_path, display_images=False):
    try:
        analyzer.img = cv2.imread(img_path)

        if display_images:
            cv2.imshow('image', resize_with_aspectratio(image=analyzer.img, width=400))
            cv2.waitKey(0)


        measurements = analyzer.resting_symmetry()

        return measurements

    except AssertionError:
        return None

def create_file():
    df_measurements = pd.DataFrame({ 'category': [], 'identifier': [], 'movement': []})

    # Handle normals 
    for normal_folder in glob.glob(os.path.join(DATA_PATH_NORMAL, '*')):
        file_identifier = os.path.basename(normal_folder)

        # For now, only calculate values for faces in rest position.
        # These are annotated in each folder as XX_1.jpg.
        image_path = os.path.join(normal_folder, file_identifier + '_1.jpg')

        result = analyze_img(image_path)

        if result is None:
            continue

        row = {
            'category': 'Normal',
            'identifier': file_identifier,
            'movement': 'relaxed',
        }

        # Copy metrics to the new row
        for key, value in result.items():
            row[key.name] = value


        df_measurements = df_measurements.append(row, ignore_index=True)


    # Handle flaccids
    for flaccid_folder in glob.glob(os.path.join(DATA_PATH_FLACCID, '*')):
        flaccid_category = os.path.basename(flaccid_folder)

        for flaccid_category_instance in glob.glob(os.path.join(flaccid_folder, '*')):
            file_identifier = os.path.basename(flaccid_category_instance)

            image_path = os.path.join(flaccid_category_instance, file_identifier + '_1.jpg')

            result = analyze_img(image_path)

            if result is None:
                continue

            row = {
                'category': flaccid_category,
                'identifier': file_identifier,
                'movement': 'relaxed'
            }

            # Copy metrics to the new row
            for key, value in result.items():
                row[key.name] = value
            
            df_measurements = df_measurements.append(row, ignore_index=True)

    df_measurements.to_csv(CSV_MEASUREMENTS_PATH)

def process_file():
    print('test')
    return

if __name__ == '__main__':
    if not os.path.exists(CSV_MEASUREMENTS_PATH):
        create_file()
    elif not os.path.exists(CSV_PROCESSED_PATH):
        process_file()