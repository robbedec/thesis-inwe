import argparse
import cv2
import os

from src.keypoints.detectors.MediapipeKPDetector import MediapipeKPDetector

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True)
parser.add_argument('-n', '--name', required=True)
args = parser.parse_args()

VIDEO_PATH = args.path
VIDEO_NAME = args.name
OUTPUT = os.path.join(VIDEO_PATH, VIDEO_NAME)

print(VIDEO_PATH, VIDEO_NAME)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT, fourcc, 20.0, (640,  480))

detector = MediapipeKPDetector()
  
while(True):
      
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    frame_cpy = frame.copy()
    out.write(frame)

    img, faces = detector.findFaceMesh(img=frame, draw_points=False, pose_estimation=True)
    angles = detector._orientation_angles

    if len(angles) == 0:
        print('No face detected')
        continue

    x, y, z = angles[0]

    p1 = (int(faces[0][1][0]), int(faces[0][1][1]))
    p2 = (int(faces[0][1][0] + y * 10) , int(faces[0][1][1] - x * 10))
    
    cv2.line(frame_cpy, p1, p2, (255, 0, 0), 3)

    border_color = (0, 255, 0) if -20 < x < 20 and -10 < y < 10 else (0, 0, 255) 

    dst = cv2.copyMakeBorder(frame_cpy, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

    # Display the resulting frame
    cv2.imshow('frame', dst)

    if cv2.waitKey(1) == 27:
        break

# Save video

# After the loop release the cap object
cap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()