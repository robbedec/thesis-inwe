import pandas as pd
import os
import glob
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from src.analysis.analyzer import StaticAnalyzer
from src.utils.util import resize_with_aspectratio
from src.analysis.enums import Measurements, MEEIMovements

DATA_PATH = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set'
DATA_PATH_NORMAL = os.path.join(DATA_PATH, 'Normals')
DATA_PATH_FLACCID = os.path.join(DATA_PATH, 'Flaccid')
DATA_PATH_IMAGES = '/home/robbedec/repos/ugent/thesis-inwe/src/images'

CSV_MEASUREMENTS_PATH = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/meei_measurements.csv'
CSV_PROCESSED_PATH = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/meei_measurements_processed.csv'

analyzer = StaticAnalyzer(draw=False)

def analyze_img(img_path, display_images=False):
    try:
        analyzer.img = cv2.imread(img_path)

        if display_images:
            cv2.imshow('image', resize_with_aspectratio(image=analyzer.img, width=400))
            cv2.waitKey(0)

        measurements = analyzer.resting_symmetry()

        return measurements

    except:
        print(img_path)
        return None

def create_file():
    df_measurements = pd.DataFrame({ 'category': [], 'identifier': [], 'movement': []})

    # Handle normals 
    for normal_folder in glob.glob(os.path.join(DATA_PATH_NORMAL, '*')):
        file_identifier = os.path.basename(normal_folder)

        # For now, only calculate values for faces in rest position.
        # These are annotated in each folder as XX_1.jpg.
        for i in range(8):
            image_path = os.path.join(normal_folder, file_identifier + '_' + str(i + 1) + '.jpg')

            result = analyze_img(image_path)

            if result is None:
                continue

            row = {
                'category': 'Normal',
                'identifier': file_identifier,
                'movement': MEEIMovements(i).name.lower()
            }

            # Copy metrics to the new row
            for key, value in result.items():
                row[key.name] = value

            df_measurements = df_measurements.append(row, ignore_index=True)

    #Handle non dataset images
    for image in ['obama.jpg', 'clooney.jpeg']:
        file_identifier = image
        image_path = os.path.join(DATA_PATH_IMAGES, image)

        result = analyze_img(image_path)

        if result is None:
            continue

        row = {
            'category': 'Normal',
            'identifier': file_identifier + str(i),
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

            for i in range(8):
                image_path = os.path.join(flaccid_category_instance, file_identifier + '_' + str(i + 1) + '.jpg')

                result = analyze_img(image_path)

                if result is None:
                    continue

                row = {
                    'category': flaccid_category,
                    'identifier': file_identifier,
                    'movement': MEEIMovements(i).name.lower() 
                }

                # Copy metrics to the new row
                for key, value in result.items():
                    row[key.name] = value
                
                df_measurements = df_measurements.append(row, ignore_index=True)

    df_measurements.to_csv(CSV_MEASUREMENTS_PATH)

def process_file():
    df_measurements = pd.read_csv(CSV_MEASUREMENTS_PATH, index_col=0)
    df_grouped = df_measurements.groupby('category')

    df_grouped_mean = df_grouped.mean()
    df_grouped_mean['aggregation_op'] = 'mean'

    df_grouped_median = df_grouped.median()
    df_grouped_median['aggregation_op'] = 'median'

    # TODO: drop columns that cannot be aggregated, will throw error in the future
    df_harmonic_mean = df_grouped.agg(lambda x: stats.hmean(x))
    df_harmonic_mean['aggregation_op'] = 'harmonic_mean'

    df_result = pd.concat([df_grouped_mean, df_grouped_median, df_harmonic_mean])

    df_result.to_csv(CSV_PROCESSED_PATH)

def plot_results(boxplot=False, movements=[]):
    df_measurements = pd.read_csv(CSV_MEASUREMENTS_PATH, index_col=0)

    if len(movements) != 0:
        df_measurements = df_measurements[df_measurements['movement'].isin(movements)]

    plot_kind = 'swarm' if not boxplot else 'box'

    measurement_categories = [e_cat.name for e_cat in Measurements]
    fig, axs = plt.subplots(len(measurement_categories), 1)

    for i, cat in enumerate(measurement_categories):
        axs[i] = sns.catplot(x='category', y=cat, data=df_measurements, kind=plot_kind)
        axs[i].set_xticklabels(rotation=45)

        plt.gcf().subplots_adjust(bottom=0.3)

        # Maybe maybe
        if boxplot:
            axs[i].set(ylim=(0, 1))

    plt.show()

if __name__ == '__main__':
    if not os.path.exists(CSV_MEASUREMENTS_PATH):
        create_file()
    elif not os.path.exists(CSV_PROCESSED_PATH):
        process_file()
    else:
        print('Measurements and processing is already available.\nPlotting results.')
        print(Measurements.EYE_DROOP.name.lower())

        # Only plot relaxed face position
        # plot_results(boxplot=False, movements=[ x.name.lower() for x in [MEEIMovements.RELAXED] ])

        # Plot all movements
        plot_results(boxplot=True)