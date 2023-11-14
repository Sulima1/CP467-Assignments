import cv2 as cv
import numpy as np

def faceRecognition(faceInput, faceGrey):
    
    #haarcascade classifiers for facial features
    faceCascade = cv.CascadeClassifier("Assignment 4\HaarCascadeFiles\haarcascade_frontalface_default.xml")
    eyeCascade = cv.CascadeClassifier("Assignment 4\HaarCascadeFiles\haarcascade_eye.xml")

    face = faceCascade.detectMultiScale(faceGrey, 1.3, 5)

    #loop for face recognition
    for (x, y, w, h) in face:
        faceInput = cv.rectangle(faceInput, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(faceInput, 'Face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        roiGrey = faceGrey[y:y+h, x:x+w]
        roiColor = faceInput[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roiGrey)
        #loop for eye recognition in each face
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv.putText(roiColor, 'Eye', (ex, ey), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv.imwrite("Assignment 4/outputs/faceOuput.png", faceInput)

    return

def carRecognition(carInput, carGrey):

    return

def main():

    #question 1.1
    faceInput = cv.imread("Assignment 4\images\objectDetection1.jpg")
    faceGrey = cv.cvtColor(faceInput, cv.COLOR_BGR2GRAY) 

    faceRecognition(faceInput, faceGrey)

    #question 1.2
    carInput = cv.imread("Assignment 4\images\objectDetection2.jpg")
    carGrey =  cv.imread(carInput, cv.COLOR_BGR2GRAY)

    carRecognition(carInput, carGrey)

if __name__ == '__main__':
    main()