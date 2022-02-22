import sys
import cv2
import time

from src.keypoints.detectors.MediapipeKPDetector import MediapipeKPDetector
from src.keypoints.detectors.MeeShapeDetector import MeeShapeDetector

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def main():
    video_source = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1.mp4'
    benchmark_active = len(sys.argv) == 2 and sys.argv[1] == '--benchmark'

    cap = cv2.VideoCapture(video_source)

    mpDetector = MediapipeKPDetector(maxFaces=1)
    meeDetector = MeeShapeDetector()

    if benchmark_active:
        import pandas as pd
        df_benchmark = pd.DataFrame({ 'mediapipe': [], 'meeshape': []})


    while True:
        success, img = cap.read()

        # Success is False when the video is in the last frame
        if not success and benchmark_active:
            df_benchmark.to_csv('MP_MEE_sidebyside.csv')

        start_time = time.time()
        img_mp, faces = mpDetector.findFaceMesh(img.copy(), filtered=True)
        end_time_mp = (time.time() - start_time) * 1000
        # print('--- Mediapipe benchmark: {} ms'.format((time.time() - start_time) * 1000))


        start_time = time.time()
        img_mee, results = meeDetector.find_keypoints(img)
        end_time_mee = (time.time() - start_time) * 1000
        #print('--- MEE Shape benchmark: {} ms'.format((time.time() - start_time) * 1000))

        resized_width = 500
        resize_mp = ResizeWithAspectRatio(img_mp, width=resized_width) 
        resize_mee = ResizeWithAspectRatio(img_mee, width=resized_width) 

        cv2.imshow("Mediapipe resized", resize_mp)
        cv2.imshow("MEEShape resized", resize_mee)

        if benchmark_active:
            df_benchmark = df_benchmark.append({ 'mediapipe': end_time_mp, 'meeshape': end_time_mee }, ignore_index=True)

        cv2.waitKey(1)
    
if __name__ == '__main__':
    main()