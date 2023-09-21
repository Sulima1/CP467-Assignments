import cv2 as cv
import numpy as np

#Image Interpolation
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

imageHeight = image.shape[0]
imageWidth = image.shape[1]

upHeight = imageHeight * 4
upWidth = imageWidth * 4
resizedDimensions = (upHeight, upWidth)

#Nearest neighbor interpolation

NNImage = cv.resize(image, resizedDimensions, cv.INTER_NEAREST)

cv.imwrite("Assignment 1\outputs\cameraman_nearest.tif", NNImage)

#Bilinear interpolation

BilinearImage = cv.resize(image, resizedDimensions, cv.INTER_LINEAR)

cv.imwrite("Assignment 1\outputs\cameraman_bilinear.tif", BilinearImage)

#Bicubic interpolation

BicubicImage = cv.resize(image, resizedDimensions, cv.INTER_CUBIC)

cv.imwrite("Assignment 1\outputs\cameraman_bicubic.tif", BicubicImage)

