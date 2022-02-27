import pandas as pd
import cv2

from src.analysis.StaticAnalyzer import StaticAnalyzer
from src.analysis.measurements import Measurements, FlaccidCategories
from src.keypoints.detectors.MediapipeKPDetector import FaceRegion, MediapipeKPDetector
from src.utils.util import resize_with_aspectratio

CSV_PROCESSED_PATH = '/home/robbedec/repos/ugent/thesis-inwe/src/analysis/csv/meei_measurements_processed.csv'
AGG_OP = 'mean'

def color_mapping():
    category_color_mapping = dict()

    category_color_mapping['Normal'] = (0, 255, 0)
    category_color_mapping['NearNormalFlaccid'] = (0, 255, 0)

    category_color_mapping['MildFlaccid'] = (0, 255, 255)
    category_color_mapping['ModerateFlaccid'] = (0, 255, 255)

    category_color_mapping['SevereFlaccid'] = (0, 0, 255)
    category_color_mapping['CompleteFlaccid'] = (0, 0, 255)

    return category_color_mapping

def draw_keypoints(img, mapping, keypoints):
    color_map = color_mapping()

    mediapipe_indices = MediapipeKPDetector.get_68KP_indices(as_dict=True)

    # Eyes
    for point_index in sum([mediapipe_indices[FaceRegion.LEFT_EYE], mediapipe_indices[FaceRegion.RIGHT_EYE]], []):
        (category, value) = mapping[Measurements.EYE_DROOP.name]

        cv2.circle(img=img, center=keypoints[point_index], radius=5, color=color_map[category], thickness=cv2.FILLED)
    
    # Mouth
    for point_index in sum([mediapipe_indices[FaceRegion.INSIDE_LIP], mediapipe_indices[FaceRegion.LOWER_LIP], mediapipe_indices[FaceRegion.UPPER_LIP]], []):
        (category1, value1) = mapping[Measurements.MOUTH_AREA.name]
        (category2, value2) = mapping[Measurements.LIPCENTER_OFFSET.name]


        compared_cat_val = round((FlaccidCategories[category1].value * value1 + FlaccidCategories[category2].value * value2) / 2)
        compared_cat_name = FlaccidCategories(compared_cat_val).name

        cv2.circle(img=img, center=keypoints[point_index], radius=5, color=color_map[compared_cat_name], thickness=cv2.FILLED)

    # Eyebrow
    for point_index in sum([mediapipe_indices[FaceRegion.LEFT_EYEBROW], mediapipe_indices[FaceRegion.RIGHT_EYEBROW]], []):
        (category1, value1) = mapping[Measurements.EYEBROW_INTERCEPT_DISTANCE.name]
        (category2, value2) = mapping[Measurements.EYEBROW_EYE_DISTANCE.name]
        (category3, value3) = mapping[Measurements.EYEBROW_HORIZONTAL_DISTANCE.name]


        compared_cat_val = round((FlaccidCategories[category1].value * value1 + FlaccidCategories[category2].value * value2 + FlaccidCategories[category3].value * value3) / 3)
        compared_cat_name = FlaccidCategories(compared_cat_val).name

        cv2.circle(img=img, center=keypoints[point_index], radius=5, color=color_map[compared_cat_name], thickness=cv2.FILLED)

    

    return img
    

def main():
    """
    Conclusie: Goed in extreme gevallen herkennen, shaky als het redelijk close is
    """

    df_processed = pd.read_csv(CSV_PROCESSED_PATH)
    df_processed = df_processed[df_processed['aggregation_op'] == AGG_OP]

    analyzer = StaticAnalyzer(draw=False)
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/src/images/obama.jpg')
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/SevereFlaccid/SevereFlaccid2/SevereFlaccid2_1.jpg')
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/SevereFlaccid/SevereFlaccid1/SevereFlaccid1_1.jpg')
    analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/MildFlaccid/MildFlaccid5/MildFlaccid5_1.jpg')
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid2/CompleteFlaccid2_1.jpg')
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid2/CompleteFlaccid2_6.jpg')

    resting_prediction = analyzer.resting_symmetry()
    img = analyzer.img
    keypoint = analyzer.keypoints

    print('\nSymmetry results:\n')
    res = dict()
    for key, value in resting_prediction.items():
        df_closest = df_processed.iloc[(df_processed[key.name] - value).abs().argsort()]

        cat = df_closest.iloc[0]['category']
        res[key.name] = (cat, value)

        print('%s: %.3f => Result: %s' % (key.name, value, cat))

    

    cv2.imshow('original', img)
    cv2.imshow('colored', draw_keypoints(img.copy(), res, keypoint))
    cv2.waitKey(0)

    #print(df_processed.head(20))

if __name__ == '__main__':
    main()