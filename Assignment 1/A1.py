import cv2 as cv
import numpy as np

"""
-----------------------------------
Image Interpolation
-----------------------------------
"""
#1

image = cv.imread("Assignment 1\Images\cameraman.tif")

#grab image dimensions
imageHeight = image.shape[0]
imageWidth = image.shape[1]

#downscale dimensions
downHeight = imageHeight // 2
downWidth = imageWidth // 2
resizedDimensions = (downHeight, downWidth)

resizedImage = cv.resize(image, resizedDimensions)

#output resized image
cv.imwrite("Assignment 1\outputs\cameraman_rescaled.tif", resizedImage)


#2 

image = cv.imread("Assignment 1\outputs\cameraman_rescaled.tif")

image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

imageHeight = image.shape[0]
imageWidth = image.shape[1]

upHeight = imageHeight * 2
upWidth = imageWidth * 2
resizedDimensions = (upHeight, upWidth)

#Nearest neighbor interpolation

#scaling constants (scale = desiredSize / startingSize) 

xScale = upWidth / (imageWidth - 1)
yScale = upHeight / (imageHeight - 1)

#matrix of 0s to be filled with pixel values
outputMatrix = np.zeros((upWidth, upHeight))

print(image[1,1])
for i in range(0, upHeight - 1):
    for j in range(0, upWidth - 1):
        outputMatrix[i+1, j+1] = image[1 + int(np.round(i / xScale)), 1 + int(np.round(j / yScale))]


#Write the output matrix to image

cv.imwrite("Assignment 1\outputs\cameraman_nearest.png", outputMatrix)


#Bilinear interpolation

BilinearImage = cv.resize(image, resizedDimensions, cv.INTER_LINEAR)

cv.imwrite("Assignment 1\outputs\cameraman_bilinear.tif", BilinearImage)

#Bicubic interpolation

BicubicImage = cv.resize(image, resizedDimensions, cv.INTER_CUBIC)

cv.imwrite("Assignment 1\outputs\cameraman_bicubic.tif", BicubicImage)


"""
-----------------------------------
Point Operations
-----------------------------------
"""

#1 

image = cv.imread("Assignment 1\Images\cameraman.tif")

#conert image to array of pixels
pixelArr = np.array(image)

#subtract each pixle value from maximum to get negative
pixelArr = 255 - pixelArr

cv.imwrite("Assignment 1\outputs\cameraman_negative.tif", pixelArr)


#2 

image = cv.imread("Assignment 1\outputs\cameraman_negative.tif")

#convert the image into a np array again
pixelArr = np.array(image)

#apply power transformation operation
pixelArr = 2 * (pixelArr**1)

cv.imwrite("Assignment 1\outputs\cameraman_power.tif", pixelArr)
