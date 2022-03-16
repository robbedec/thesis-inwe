from math import dist, gamma
from cv2 import phase
from matplotlib.pyplot import draw
import numpy as np
from utils.util import orthogonal_projection, resize_with_aspectratio, normalize_uint8
import cv2
from src.analysis.analyzer import StaticAnalyzer

"""
Sanity check: look if measurements are approx equal if image is rotated 90 degrees.

MOUTH_AREA seems to contain a bug (TODO). Bugs stays even when using shapely lib to
calculate the area of the polygon.

Caused by miscalculations by the mediapipe model (mouth corners)
"""
"""
test = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1_8.jpg')
test_rotated = cv2.rotate(test, cv2.ROTATE_90_CLOCKWISE)

x = StaticAnalyzer(img=test, draw=True)
x.resting_symmetry(print_results=True)

print()

x.img = test_rotated
x.resting_symmetry(print_results=True)

cv2.imshow('normal', test)
cv2.imshow('rotated', test_rotated)
cv2.waitKey(0)
"""

"""
Gabor Filter test
"""

# afbeelding met neusplooi aan beide kanten
#img_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/MildFlaccid/MildFlaccid1/MildFlaccid1_1.jpg'
img_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Normals/Normal1/Normal1_5.jpg'

# Zeer goeie afbeelding met theta = 3PI/4
#img_path = '/home/robbedec/repos/ugent/thesis-inwe/src/images/paralysis_test.jpg'

# afbeelding zonder neusplooi
img_path = '/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid2/CompleteFlaccid2_1.jpg'

img = cv2.imread(filename=img_path)
img = resize_with_aspectratio(image=img, width=400)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Change orientation with 3*PI/4
# Goeie 
m = 15
kernel = cv2.getGaborKernel(ksize=(m, m), sigma=5, theta=np.pi/4, lambd=np.pi/4, gamma=0.1, psi=1)
kernel1 = cv2.getGaborKernel(ksize=(m, m), sigma=5, theta=np.pi/4, lambd=10, gamma=0.5)

kernel_8U = normalize_uint8(kernel)
kernel_resized = cv2.resize(kernel, (400, 400))


filtered = cv2.filter2D(src=img_gray, ddepth=cv2.CV_8UC3, kernel=kernel)
filtered = normalize_uint8(filtered)

cv2.imshow('Original', img)
cv2.imshow('Kernel', kernel_resized)
cv2.imshow('GABOR', filtered)
#cv2.waitKey(0)

# Try with DOG
middle_index = m // 2
square = np.zeros(shape=(m,m))
gaussian_high_std = cv2.getGaussianKernel(ksize=m, sigma=5)
square[:, middle_index] = gaussian_high_std.T

# Create another 1D gaussian (row vector) and use it as a filter on the
# square matrix that contains the first gaussian as a column.
gaussian_low_std = cv2.getGaussianKernel(ksize=m, sigma=0.5).T
square = cv2.filter2D(src=square, kernel=gaussian_low_std, ddepth=-1)

dog_vertical = cv2.Sobel(src=square, ddepth=-1, dx=1, dy=0, ksize=3)

# Yellow line are approx 75 degrees, to orientate the DoG filter (vertical
# one is standing at 90 degrees) we rotate it 15 degrees counter clockwise.
rotation_matrix = cv2.getRotationMatrix2D(center=(middle_index, middle_index), angle=-45, scale=1)

rotated_dog_vertical = cv2.warpAffine(src=dog_vertical, M=rotation_matrix, dsize=(m, m))
img_gray_filtered = cv2.filter2D(src=img_gray/255, ddepth=cv2.CV_32F, kernel=rotated_dog_vertical)
img_gray_filtered = np.abs(img_gray_filtered)
img_gray_filtered = normalize_uint8(img_gray_filtered)

cv2.imshow('DOG', img_gray_filtered)
cv2.waitKey(0)