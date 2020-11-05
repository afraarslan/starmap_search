import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv2
#import matplotlib.pyplot as plt
import os


img1 = cv2.imread('Small_area.png',0)          # queryImage
img2 = cv2.imread('StarMap.png',0) # trainImage
