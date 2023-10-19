import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def createGaussianKernel(sigma, filterShape):
    
    h, w = filterShape
    halfHeight = h//2
    halfWidth =  w//2

    guassianFilter = np.zeros((w, h), np.float32)

    for x in range(-halfHeight, halfHeight):
        for y in range(-halfWidth, halfWidth):
            normal = 1 / (2 * np.pi * sigma**2.0)
            exponent = np.exp((x**2 + y**2) / (2 * sigma**2))
            guassianFilter[x+halfHeight, y+halfWidth] = normal*exponent


def guassianConvolution(image, kernel)






