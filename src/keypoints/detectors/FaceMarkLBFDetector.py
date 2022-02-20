import cv2
import numpy as np
import random

class FaceMarkLBFDetector:

    def __init__(self):
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel('./models/lbfmodel.yaml')
        self.cascade = cv2.CascadeClassifier('./models/lbpcascade_frontalface_improved.xml')

    def find_keypoints(self, img):
        faces = self.cascade.detectMultiScale(img, 1.05, 3, cv2.CASCADE_SCALE_IMAGE, (30, 30))
        ok, landmarks = self.facemark.fit(img, faces=faces)

        for marks in landmarks:
            cv2.face.drawFacemarks(img, marks, (255, 0, 0))

        return img

def main():
    use_video = False

    # Load the resource
    # cap = cv2.VideoCapture("Videos/1.mp4")
    cap = cv2.VideoCapture(0) if use_video else cv2.imread('../../images/paralysis_test.jpg') 
    pTime = 0
    detector = FaceMarkLBFDetector()

    while True:
        if use_video:
            success, img = cap.read()
        else:
            img = cap

        img = detector.find_keypoints(img)

        cv2.imshow("Image", img)

        if use_video:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)
            break

if __name__ == '__main__':
    main()