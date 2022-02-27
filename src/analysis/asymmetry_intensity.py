import pandas as pd
import cv2

from src.analysis.StaticAnalyzer import StaticAnalyzer
from src.analysis.measurements import Measurements
from src.keypoints.detectors.MediapipeKPDetector import FaceRegion
from src.utils.util import resize_with_aspectratio

CSV_PROCESSED_PATH = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/meei_measurements_processed.csv'
AGG_OP = 'mean'

def color_mapping():
    category_color_mapping = dict()

    category_color_mapping['Normal'] = (0, 255, 0)
    category_color_mapping['NearNormal'] = (0, 255, 0)

    category_color_mapping['MildFlaccid'] = (0, 255, 255)
    category_color_mapping['ModerateFlaccid'] = (0, 255, 255)

    category_color_mapping['SevereFlaccid'] = (0, 0, 255)
    category_color_mapping['CompleteFlaccid'] = (0, 0, 255)

    return category_color_mapping

def draw_keypoints(img, mapping):
    color_mapping = color_mapping()

def main():
    """
    Conclusie: Goed in extreme gevallen herkennen, shaky als het redelijk close is
    """

    df_processed = pd.read_csv(CSV_PROCESSED_PATH)
    df_processed = df_processed[df_processed['aggregation_op'] == AGG_OP]

    analyzer = StaticAnalyzer(draw=True)
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/src/images/clooney.jpeg')
    analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/SevereFlaccid/SevereFlaccid2/SevereFlaccid2_1.jpg')
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/SevereFlaccid/SevereFlaccid1/SevereFlaccid1_1.jpg')
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/MildFlaccid/MildFlaccid5/MildFlaccid5_1.jpg')

    resting_prediction = analyzer.resting_symmetry()
    img = analyzer.img

    print('\nSymmetry results:\n')
    res = dict()
    for key, value in resting_prediction.items():
        df_closest = df_processed.iloc[(df_processed[key.name] - value).abs().argsort()]

        cat = df_closest.iloc[0]['category']
        res[key.name] = cat

        print('%s: %.3f => Result: %s' % (key.name, value, cat))

    

    cv2.imshow('test', img)
    cv2.waitKey(0)

    #print(df_processed.head(20))

if __name__ == '__main__':
    main()