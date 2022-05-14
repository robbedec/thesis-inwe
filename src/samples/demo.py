import numpy as np
import cv2

from src.analysis.analyzer import StaticAnalyzer
from src.keypoints.detectors.MediapipeKPDetector import MediapipeKPDetector
from src.utils.util import resize_with_aspectratio

VIDEO_PATH = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid2/CompleteFlaccid2.mp4'
OUTPUT_VIDEO_PATH = '/media/robbedec/USB HERMES/masterproef/robbe.avi'

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 60, (width*3, height))
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 60, (1920, 1137))

# Init analyzers
analyzer = StaticAnalyzer(draw=True, mp_static_mode=False)
detector = MediapipeKPDetector(staticMode=False)

count = 0

while(True):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #frame = resize_with_aspectratio(frame, width=400)

    analyzer.img = frame.copy()
    analyzer.resting_symmetry(print_results=False)
    
    im_raw = frame
    im_keypoints, faces = detector.findFaceMesh(frame.copy(), draw_points=True, draw_indices=False, filtered=False, pose_estimation=False)
    im_annotations = analyzer.img

    for im, text in zip([im_raw, im_keypoints, im_annotations], ['Raw video', 'Detected keypoints', 'Symmetry annotations']):
        #cv2.putText(im, text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
        cv2.putText(im, text, (20, height - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,), 7)
        cv2.putText(im, text, (20, height - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 4)

    im_stacked = np.hstack([im_raw, im_keypoints, im_annotations])
    im_stacked = resize_with_aspectratio(im_stacked, width=1920)
    out.write(im_stacked)
    #cv2.imshow('Demo', im_stacked)

    if cv2.waitKey(1) == 27:
        break

# After the loop release the cap object
cap.release()
out.release()
cv2.destroyAllWindows()