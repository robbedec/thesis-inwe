from dlib import get_frontal_face_detector
from dlib import shape_predictor
from dlib import rectangle

import cv2
import numpy as np

class MeeShapeDetector():

    def __init__(self, predictor_path='/home/robbedec/repos/ugent/thesis-inwe/notebooks/facial_keypoint/detectors/models/mee_shape_predictor_68_face_landmarks.dat'):

        self.face_detector = get_frontal_face_detector()
        self.predictor = shape_predictor(predictor_path)
        self.results = []

    def find_keypoints(self, img, speed_transform=True, draw=True):
        self.results = []
        height, width, d = img.shape                        

        # Convert image to grey scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if d > 1 else img

        # Resize to speed up face dectection
        if speed_transform:
            newWidth=200
            ScalingFactor=width/newWidth
            newHeight=int(height/ScalingFactor)

            img_small = cv2.resize(img_gray, (newWidth, newHeight), interpolation=cv2.INTER_AREA)

        faces = self.face_detector(img_small if speed_transform else img_gray)

        # The image resizing may cause undetected faces.
        # Retry with the original size
        if len(faces) == 0:
            faces = self.face_detector(img_gray)
        
        # assert len(faces) != 0

        for (i, rect) in enumerate(faces):
            # print('Detected face nr {}'.format(i))

            keypoints_shape = np.zeros((68,2),dtype=int)

            #adjust face position using the scaling factor
            mod_rect=rectangle(
                    left=int(rect.left() * ScalingFactor), 
                    top=int(rect.top() * ScalingFactor), 
                    right=int(rect.right() * ScalingFactor), 
                    bottom=int(rect.bottom() * ScalingFactor)
            )

            #predict facial landmarks 
            shape_dlib = self.predictor(img, mod_rect)   
            #shape_dlib = predictor(gray, rect) 
        
            #transform shape object to np.matrix type
            for k in range(0,68):
                
                keypoints_shape[k] = (shape_dlib.part(k).x, shape_dlib.part(k).y)
                
                if keypoints_shape[k,0]<= 0: 
                    keypoints_shape[k,0] = 1
                    
                if keypoints_shape[k,1]<= 0: 
                    keypoints_shape[k,1] = 1

            #position of the face in the image
            _boundingbox=[int(rect.left() * ScalingFactor), 
                            int(rect.top() * ScalingFactor),
                            int(rect.right() * ScalingFactor) - int(rect.left() * ScalingFactor),
                            int(rect.bottom() * ScalingFactor) - int(rect.top() * ScalingFactor)]
            
            self.results.append(np.copy(keypoints_shape))
        
        if draw:
            img = self.draw_keypoints(img.copy())

        return img, self.results 
    
    def draw_keypoints(self, img, contour=False):
        h, w, _ = img.shape
        circle_radius = 2 if h < 1000 else 6

        for i, val in enumerate(self.results):
            for j, (x, y) in enumerate(val):
                #print(j, x, y)
                if not contour and j < 17:
                    continue

                cv2.circle(img, (x,y), circle_radius, (255,0,0), cv2.FILLED)

        return img

# Sample usage
def main():
    use_video = True

    # Load the resource
    # cap = cv2.VideoCapture("Videos/1.mp4")
    cap = cv2.VideoCapture(0) if use_video else cv2.imread('../../images/paralysis_test.jpg') 
    pTime = 0
    detector = MeeShapeDetector(predictor_path='./models/mee_shape_predictor_68_face_landmarks.dat')

    while True:
        if use_video:
            success, img = cap.read()
        else:
            img = cap

        # Call landmark generator
        # Results contains a (68,2) np arry for each detected face
        img, results = detector.find_keypoints(img)

        cv2.imshow("Image", img)

        if use_video:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)
            break
    
if __name__ == '__main__':
    main()