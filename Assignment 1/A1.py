import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

imageGrey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

imageHeight = image.shape[0]
imageWidth = image.shape[1]

upHeight = imageHeight * 2
upWidth = imageWidth * 2
resizedDimensions = (upHeight, upWidth)

#Nearest neighbor interpolation
1
#scaling constants (scale = desiredSize / startingSize) 

xScale = upWidth / (imageWidth - 1)
yScale = upHeight / (imageHeight - 1)

#matrix of 0s to be filled with pixel values
outputMatrix = np.zeros((upWidth, upHeight))

for i in range(0, upHeight - 1):
    for j in range(0, upWidth - 1):
        outputMatrix[i+1, j+1] = imageGrey[1 + int(np.round(i / xScale)), 1 + int(np.round(j / yScale))]


#Write the output matrix to image

cv.imwrite("Assignment 1\outputs\cameraman_nearest.png", outputMatrix)


#Bilinear interpolation




#high level interpolation
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

#convert the image into a np array again
pixelArr = np.array(image)

#apply power transformation operation
pixelArr = 2 * (pixelArr**1.15).astype(np.uint8)

cv.imwrite("Assignment 1\outputs\cameraman_power.tif", pixelArr)

#3

minPixelVal = np.min(image)
maxPixelVal = np.max(image)

imageArr = np.array(image)

contrastArr = ((imageArr - minPixelVal) / (maxPixelVal - minPixelVal))*255


cv.imwrite("Assignment 1\outputs\cameraman_contrast.png", contrastArr)

"""
------------------------------------
Histogram Processing
------------------------------------
"""
#1

image = cv.imread("Assignment 1\Images\Einstein.tif")

einsteinGrey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

#retrieve histogram and bin value of greyscale image
histogram, bins = np.histogram(einsteinGrey.flatten(), 256, [0,256])

#calculate the cumulative sum of the histogram
cumulativeSum = histogram.cumsum()
#normalize cumulative sum
normalizedCum = cumulativeSum * histogram.max()/cumulativeSum.max()

#create a masked array to perform operations
cumalativeSumMask = np.ma.masked_equal(cumulativeSum,0)
cumalativeSumMask = (cumalativeSumMask - cumalativeSumMask.min())*255/(cumalativeSumMask.max()-cumalativeSumMask.min())
cumulativeSum = np.ma.filled(cumalativeSumMask,0)

equalizedImage = cumulativeSum[einsteinGrey]

cv.imwrite("Assignment 1\outputs\Einstein_equalized.png", equalizedImage)

#2

image = cv.imread("Assignment 1\Images\Chest_x-ray1.jpeg", cv.IMREAD_GRAYSCALE)

referenceImage = cv.imread("Assignment 1\Images\Chest_x-ray2.jpeg", cv.IMREAD_GRAYSCALE)

sourceHistogram, bins = np.histogram(image.flatten(), 256, [0,256])
sourceCdf = sourceHistogram.cumsum()


referenceHistogram, bins = np.histogram(referenceImage.flatten(), 256, [0,256])
referenceCdf = referenceHistogram.cumsum()


pixels = np.arange(256)

newPixels = np.interp(sourceCdf, referenceCdf, pixels)
imageMatch = (np.reshape(newPixels[image.ravel()], image.shape)).astype(np.uint8)

cv.imwrite("Assignment 1\outputs\chest_x-ray3.png", imageMatch)

