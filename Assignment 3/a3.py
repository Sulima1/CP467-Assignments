import cv2 as cv
import numpy as np

def detectCircle(irisImage, minRadius, cannyThreshold, centreThreshold):
    #apply median blur to reduce image noise
    irisImage = cv.medianBlur(irisImage, 7)

    rows = irisImage.shape[0]
    
    #apply the Hough Circle Transformation
    irisCircle = cv.HoughCircles(irisImage, cv.HOUGH_GRADIENT, 1, rows / 8, param1=cannyThreshold, param2=centreThreshold, minRadius=minRadius, maxRadius=0)

    irisOutput = irisImage.copy()

    #draw the detected circles if found
    if irisCircle is not None:
        irisCircle = np.uint16(np.around(irisCircle))
        for i in irisCircle[0, :]:
            center = (i[0], i[1])
            irisOutput = cv.circle(irisImage, center, 1, (255, 100, 100), 3)
            radius = i[2]
            irisOutput = cv.circle(irisImage, center, radius, (70, 255, 150), 5)
    
    return irisOutput


def main():
    #default threshold values for finding pupil
    cannyThreshold, centreThreshold = 100, 40
    #small radius for finding pupil, increase when outlining iris
    radiusList = [0,35]
    
    for minRadius in radiusList:
        for i in range(1, 6):
            irisImage = cv.imread('Assignment 3/images/' + str(i) + '.bmp', 0)

            #if looking for iris then edit thresholds
            if minRadius != 0:
                cannyThreshold, centreThreshold = 21, 100

            irisOutput = detectCircle(irisImage, minRadius, cannyThreshold, centreThreshold)
            if minRadius == 0:
                cv.imwrite('Assignment 3/TempResults/outputPupil' + str(i) + '.bmp', irisOutput)
            else:
                cv.imwrite('Assignment 3/TempResults/outputIris' + str(i) + '.bmp', irisOutput)

    #Initially I designed the code expecting to have 2 sets of outptus, iris and sclera    
    #Merge the two images together to show both circles
    for j in range(1, 6):
        pupilImage = cv.imread('Assignment 3/TempResults/outputPupil' + str(j) + '.bmp')
        irisImage = cv.imread('Assignment 3/TempResults/outputIris' + str(j) + '.bmp')
        mergedImage = cv.addWeighted(pupilImage, 0.45, irisImage, 0.45, 0.0)
        cv.imwrite('Assignment 3/Segmented_Results/' + str(j) + '_segmented.bmp', mergedImage)


if __name__ == "__main__":
    main()