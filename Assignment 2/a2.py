import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def AverageSmoothingLowLevel(image, kernel):
    
    #Apply cross correlation to our kernel 
    kernel = np.flipud(np.fliplr(kernel))

    #get variables to compute matrix size of output
    xKernel = kernel.shape[0]
    yKernel = kernel.shape[1]
    xImage = image.shape[0]
    yImage = image.shape[1]

    #Set padding an strides for kernel
    padding = 0
    strides = 1

    xOutput = int(((xImage - xKernel + 2 * padding) / strides) + 1)
    yOutput = int(((yImage - yKernel + 2 * padding) / strides) + 1)

    #create the output matrix for the image to replace
    outputImage = np.zeros((xOutput, yOutput))

    imagePadded = image

    for y in range(yImage):
        if y > yImage - yKernel:
            break
        if y % strides == 0:
            for x in range(xImage):
                if x > xImage - xKernel:
                    break

                if x % strides == 0:
                    outputImage[x, y] = (kernel * imagePadded[x: x + xKernel, y: y + yKernel]).sum()

    cv.imwrite("Assignment 2\outputs\lowlevel_average_cameraman.png", outputImage)

    return 

def GaussianSmoothingLowLevel(image, kernel, sigma, mean):
    padding = 0
    strides = 1    

    xKernel = kernel.shape[0]
    yKernel = kernel.shape[1]
    xImage = image.shape[0]
    yImage = image.shape[1]

    xOutput = int(((xImage - xKernel + 2 * padding) / strides) + 1)
    yOutput = int(((yImage - yKernel + 2 * padding) / strides) + 1)

    #create the output matrix for the image to replace
    outputImage = np.zeros((xOutput, yOutput))

    imagePadded = image

    for y in range(yImage):
        if y > yImage - yKernel:
            break
        if y % strides == 0:
            for x in range(xImage):
                if x > xImage - xKernel:
                    break

                if x % strides == 0:
                    outputImage[x, y] = (kernel * imagePadded[x:x+xKernel, y:y+yKernel]).sum()

    cv.imwrite("Assignment 2\outputs\lowlevel_gaussian_cameraman.png", outputImage)    

def SobelLowLevel(image, xSobel, ySobel):

    if image.ndim == 3:
        image = np.mean(image, axis=2)

    xGradient = np.zeros_like(image)
    yGradient = np.zeros_like(image) 

    xImage = image.shape[0]
    yImage = image.shape[1]

    outputImage = np.zeros((xImage, yImage))

    for x in range(1, xImage - 1):
        for y in range(1, yImage - 1):
            xGradient[x, y] = np.sum(image[x-1:x+2, y-1:y+2] * xSobel)
            yGradient[x, y] = np.sum(image[x-1:x+2, y-1:y+2] * ySobel)

    sharpenedImage = image + xGradient + yGradient
    #sharpenedImage = np.clip(sharpenedImage, 0, 255)

    cv.imwrite("Assignment 2\outputs\lowlevel_sobel_cameraman.png", sharpenedImage)

def main():
    #Question 1a: Average Smoothing filter
    kernel = [[1/9,1/9,1/9],
              [1/9,1/9,1/9,],
              [1/9,1/9,1/9]]
    image = cv.imread("Assignment 2\cameraman.tif")
    image = cv.cvtColor(src=image, code=cv.COLOR_BGR2GRAY)
    
    AverageSmoothingLowLevel(image, kernel)

    #Question 1b: Guassian Smoothing Filter
    sigma = 1
    mean = 0
    kernel = np.fromfunction(lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(- ((x - 3)**2 + (y - 3)**2) / (2 * sigma**2)), (7, 7))
    kernel /= np.sum(kernel)

    GaussianSmoothingLowLevel(image, kernel, sigma, mean)

    #Question 1c: Sobel Smoothing Filter
    xSobel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    ySobel = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    SobelLowLevel(image, xSobel, ySobel)

    #Question 2 high levels

    #average blur
    output = cv.blur(image, (3, 3))
    cv.imwrite("Assignment 2\outputs\highlevel_average_cameraman.png", output)

    #gaussian blur
    output = cv.GaussianBlur(image, (7, 7), 1)
    cv.imwrite("Assignment 2\outputs\highlevel_gaussian_cameraman.png", output)

    #sobel
    output = cv.Sobel(image,cv.CV_64F,1,0,ksize=3) + cv.Sobel(image,cv.CV_64F,0,1,ksize=3)
    cv.imwrite("Assignment 2\outputs\highlevel_sobel_cameraman.png", output)

    

if __name__ == '__main__':
    main()