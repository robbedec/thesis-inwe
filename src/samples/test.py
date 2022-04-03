import cv2
from src.analysis.analyzer import StaticAnalyzer

"""
Sanity check: look if measurements are approx equal if image is rotated 90 degrees.

MOUTH_AREA seems to contain a bug (TODO). Bugs stays even when using shapely lib to
calculate the area of the polygon.

Caused by miscalculations by the mediapipe model (mouth corners)
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