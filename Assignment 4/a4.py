import cv2 as cv
import numpy as np


def faceRecognition(faceInput, faceGrey):
    
    #haarcascade classifiers for facial features
    faceCascade = cv.CascadeClassifier("Assignment 4\HaarCascadeFiles\haarcascade_frontalface_default.xml")
    eyeCascade = cv.CascadeClassifier("Assignment 4\HaarCascadeFiles\haarcascade_eye.xml")

    face = faceCascade.detectMultiScale(faceGrey, 1.3, 5)

    sift = cv.SIFT_create(contrastThreshold=0.08, edgeThreshold=12)
    kp = sift.detect(faceInput,None)
    faceInput = cv.drawKeypoints(faceInput,kp,faceGrey)

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

    carCascade = cv.CascadeClassifier("Assignment 4\HaarCascadeFiles\cars.xml")

    car = carCascade.detectMultiScale(carGrey, 1.1, 1, minSize=(30, 30))

    sift = cv.SIFT_create(contrastThreshold=0.12, edgeThreshold=15)
    kp = sift.detect(carInput,None)
    carInput = cv.drawKeypoints(carInput,kp, carGrey)

    for (x, y, w, h) in car:
        carInput = cv.rectangle(carInput, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(carInput, 'Car', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv.imwrite("Assignment 4/outputs/carOutput.png", carInput) 
    
    return


def keyPointDetection(faceInput, carInput, faceGrey, carGrey):

    images = [("faceInput", faceGrey, faceInput), ("carInput", carGrey, carInput)]

    sift = cv.SIFT_create(contrastThreshold=0.08, edgeThreshold=5)

    for name, i, j in images:
        kp = sift.detect(i,None)
        j = cv.drawKeypoints(i,kp,j)

        cv.imwrite("Assignment 4/outputs/" + name + "output.png", j) 

    return

def imageManipulation(backgroundImage, figureImage):

    figureImage = cv.resize(figureImage, (figureImage.shape[0]//2, figureImage.shape[1]//2))

    # Assuming the source and destination images have the same size
    mask = 255 * np.ones(figureImage.shape, figureImage.dtype)

    # This is where the CENTER of the source image will be placed
    center = (backgroundImage.shape[1] // 2, backgroundImage.shape[0] // 2 - 250)

    # Convert BGR to HSV
    hsv = cv.cvtColor(figureImage, cv.COLOR_BGR2HSV)

    # Reduce saturation
    saturationFactor = 0.4
    hsv[:, :, 1] = hsv[:, :, 1] * saturationFactor

    # Convert back to BGR
    desaturatedImage = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Poisson blending
    result = cv.seamlessClone(desaturatedImage, backgroundImage, mask, center, cv.MIXED_CLONE)

    # Display the result
    cv.imwrite("Assignment 4/outputs/mixedImage.jpg", result) 
    

    return

def imageStitching(leftImage, leftImageGrey, rightImage, rightImageGrey):

    sift = cv.SIFT_create()
    # Find keypoints and descriptors
    keypoint1, descriptor1 = sift.detectAndCompute(leftImageGrey, None)
    keypoint2, descriptor2 = sift.detectAndCompute(rightImageGrey, None)

    # Use a FLANN based matcher to find matches
    matcher = cv.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = matcher.knnMatch(descriptor1, descriptor2, k=2)

    # Ratio test to get good matches
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    # Extract matched keypoints
    sourcePoints = np.float32([keypoint1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    destinationPoints = np.float32([keypoint2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, _ = cv.findHomography(sourcePoints, destinationPoints, cv.RANSAC, 5.0)

    # Warp leftImage to align with rightImage
    result = cv.warpPerspective(leftImage, H, (rightImage.shape[1] + leftImage.shape[1], rightImage.shape[0]))

    # Copy leftImage to the result image
    result[0:rightImage.shape[0], 0:rightImage.shape[1]] = rightImage

    cv.imwrite("Assignment 4/outputs/Panorama.jpg", result)

    return

def main():

    """
    #question 1: just SIFT if that was required, no labels
    faceInput = cv.imread("Assignment 4\images\objectDetection1.jpg")
    carInput = cv.imread("Assignment 4\images\objectDetection2.jpg")

    faceGrey = cv.cvtColor(faceInput, cv.COLOR_BGR2GRAY)
    carGrey =  cv.cvtColor(carInput, cv.COLOR_BGR2GRAY)

    keyPointDetection(faceInput, carInput, faceGrey, carGrey)
    """
    
    #question 1.1: SIFT + HaarCascade
    faceInput = cv.imread("Assignment 4\images\objectDetection1.jpg")
    faceGrey = cv.cvtColor(faceInput, cv.COLOR_BGR2GRAY) 

    faceRecognition(faceInput, faceGrey)

    #question 1.2: SIFT + HaarCascade
    carInput = cv.imread("Assignment 4\images\objectDetection2.jpg")
    carGrey =  cv.cvtColor(carInput, cv.COLOR_BGR2GRAY)

    carRecognition(carInput, carGrey)
    

    #question 2
    backgroundImage = cv.imread("Assignment 4\images\deskImage.jpg")
    figureImage = cv.imread("Assignment 4/images/figureImage.jpg")

    imageManipulation(backgroundImage, figureImage)

    #question 3
    leftImage = cv.imread("Assignment 4\images\imageStitch1.jpg")
    leftImageGrey = cv.cvtColor(leftImage, cv.COLOR_BGR2GRAY)

    rightImage = cv.imread("Assignment 4\images\imageStitch2.jpg")
    rightImageGrey = cv.cvtColor(rightImage, cv.COLOR_BGR2GRAY)

    imageStitching(leftImage, leftImageGrey, rightImage, rightImageGrey) 
    
if __name__ == '__main__':
    main()