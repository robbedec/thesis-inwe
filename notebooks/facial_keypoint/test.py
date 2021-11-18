import random
import numpy as np
import cv2 as cv


def main():
    # Read image
    start_frame = readImage('../images/robbedec.jpeg')
    frame = np.vstack((start_frame, start_frame))

    facemark = cv.face.createFacemarkLBF()
    loadModel(facemark)

    cascade = cv.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    if cascade.empty() :
        print("cascade not found")
        exit()

    faces = cascade.detectMultiScale(frame, 1.05,  3, cv.CASCADE_SCALE_IMAGE, (30, 30))
    ok, landmarks = facemark.fit(frame, faces=faces)

    # cv.imshow("Image", frame)

    drawMarks(landmarks, frame)

    cv.imshow("Image Landmarks", frame)
    cv.waitKey()

def readImage(path):
    frame = cv.imread(path)
    if frame is None:
        print("image not found")
        exit()
    
    return frame

def loadModel(facemark):
    try:
        facemark.loadModel('lbfmodel.yaml')
    except cv.error:
        print("Model not found\nlbfmodel.yaml can be download at")
        print("https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml")

def getKeypoints(marks):
    result = []

    for row in [17, 18, 19, 20, 21]:
        print(marks[:,row,:][0])
        rows = marks[:,row,:][0]
        for i in range(2):
            result.append(rows[i])

    print('\ntest\n')
    result = np.array(result)

    result = result.reshape(1,5,2)
    print(result.shape)
    return result

def drawMarks(landmarks, frame):
    for marks in landmarks:
        sampled_marks = getKeypoints(marks)
        
        couleur = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv.face.drawFacemarks(frame, sampled_marks, couleur)

if __name__=="__main__": 
    main() 