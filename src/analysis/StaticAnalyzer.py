import cv2
import sys
import numpy as np
from enum import Enum

sys.path.append('../keypoints/detectors')
sys.path.append('../utils')

from MediapipeKPDetector import MediapipeKPDetector
from util import dist_point_to_line, dist_point_to_point, mean_position, round_tuple

class Measurements(Enum):
    Eyebrows = 1

class StaticAnalyzer():

    def __init__(self, img=None):
        """
        Throws an assertion error if no faces were detected 
        """

        self._detector = MediapipeKPDetector(maxFaces=1)

        if img == None:
            # Load default image
            self.load_img(cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/src/images/obama.jpg'))
            #self.load_img(cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1_1.jpg'))
        else:
            self.load_img(img)

    def load_img(self, img):
        """
        Handle for loading in an image, this triggers the feature detection procedure
        and updates paramaters to ensure that other functions work correctly. 
        """
        self._img = img

        # Detect facial landmarks
        # Passing a copy to the detector prevents default drawings by mediapipe
        self._img_mp, self._faces = self._detector.findFaceMesh(self._img.copy(), draw_points=True, draw_indices=False, filtered=False)
        assert len(self._faces) > 0

        self._face = self._faces[0]

    def estimate_symmetry_line(self, draw=False):
        """
        Finds the equation of the line that connects the two inner corners of
        the eyes and calculates the perpendicular line that passes through the
        center of the first line 

        Returns: The slope of the line that connects both eyes and of theperpendicular line
                 together with the intersection point between both line. This provides enough
                 information to derive both equations.
        """
        print(dist_point_to_line(5, (0, 8), (1,1)))

        img_x, img_y, _ = self._img.shape

        # Draw line through the inside eye corners (points 133 & 362 for unfiltered keypoints or 39 & 42)
        (x1, y1) = self._face[133]
        (x2, y2) = self._face[362]

        # Point 168 is halfway between the eyes
        x_mid, y_mid = ((x1 + x2) / 2, (y1 + y2) / 2)

        horizontal_slope = (y2 - y1) / (x2 - x1)
        vertical_slope = np.inf if horizontal_slope == 0 else -1 / horizontal_slope


        if draw:
            # Form: y=f(x)
            horizontal_line = lambda x: (horizontal_slope) * (x - x1) + y1

            # Form x=f(y), be careful when horizontal slope = 0
            vertical_line = lambda y: x_mid if np.isinf(vertical_slope) else ((y - y_mid) / vertical_slope) + x_mid

            cv2.line(img=self._img, pt1=(0, round(horizontal_line(0))), pt2=(img_x, round(horizontal_line(img_x))), color=(0, 255, 0), thickness=1)
            cv2.line(img=self._img, pt1=(round(vertical_line(0)), 0), pt2=(round(vertical_line(img_y)), img_y), color=(0, 255, 0), thickness=1)

        return horizontal_slope, vertical_slope, (x_mid, y_mid)

    def quantify_eyebrows(self, draw=False):
        # https://pdfs.semanticscholar.org/b436/2cd87ad219790800127ddd366cc465606a78.pdf

        # Find mean coördinate in left eyebrow
        MLEB = mean_position([70, 63, 105, 66, 107], self._face)
        print(MLEB)
        
        # Find mean coördinate in right eyebrow
        MREB = mean_position([336, 296, 334, 293, 300], self._face)
        print(MREB)

        # Mean positions of the eyes (iris)
        # Approximates the center fo the eye
        LEC = mean_position([468, 469, 470, 471, 472], self._face)
        REC = mean_position([473, 474, 475, 476, 477], self._face)

        D_EB_EYE_L = dist_point_to_point(MLEB, LEC)
        D_EB_EYE_R = dist_point_to_point(MREB, REC)

        # The ratio should be 1 if both distances are equal
        # This could also be used in when moving the eyebrow as an indication that
        # one eyebrow is moving but the other one isn't.
        print(D_EB_EYE_L / D_EB_EYE_R)

        # Second metric: distance between average eyebrow points and the vertical line
        # across the eyes

        slope_h, slope_v, intercept = self.estimate_symmetry_line(draw=False)

        D_MLEB_horizontal = dist_point_to_line(slope=slope_h, slope_point=intercept, point=MLEB)
        D_MREB_horizontal = dist_point_to_line(slope=slope_h, slope_point=intercept, point=MREB)

        # Door te vergelijken met de intercept kunnen we ook meten als de wenkbrouw naar schuin
        # boven trekt of niet
        D_MLEB_intercept = dist_point_to_point(intercept, MLEB)
        D_MREB_intercept = dist_point_to_point(intercept, MREB)

        if draw:
            # Todo: option to draw all lines that were measured on the image 

            # Draw approximated eye center points
            cv2.circle(self._img, round_tuple(LEC), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(self._img, round_tuple(REC), 5, (0, 255, 0), cv2.FILLED)

            # Draw mean eyebrow points
            cv2.circle(self._img, round_tuple(MLEB), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(self._img, round_tuple(MREB), 5, (0, 255, 0), cv2.FILLED)

            # Draw lines between mean eyebrow and approximated iris
            cv2.line(img=self._img, pt1=round_tuple(LEC), pt2=round_tuple(MLEB), color=(0, 255, 0), thickness=1)
            cv2.line(img=self._img, pt1=round_tuple(REC), pt2=round_tuple(MREB), color=(0, 255, 0), thickness=1)

            # Draw lines between mean eyebrow and intercept
            cv2.line(img=self._img, pt1=round_tuple(intercept), pt2=round_tuple(MLEB), color=(0, 255, 0), thickness=1)
            cv2.line(img=self._img, pt1=round_tuple(intercept), pt2=round_tuple(MREB), color=(0, 255, 0), thickness=1)
    
    def quantify_mouth(self, draw=False):
        corner_left = self._face[61]
        corner_right = self._face[291]

        # Maybe better to use a point on the symmetry line to calculate the angles instead
        # of the endpoint of the chin.
        # For now use the projection of the chinpoint on the symmetry line
        chin = self._face[152]
        chin_corrected = 0

        if draw:
            cv2.circle(img=self._img, center=corner_left, radius=5, color=(0, 255, 0), thickness=cv2.FILLED)
            cv2.circle(img=self._img, center=corner_right, radius=5, color=(0, 255, 0), thickness=cv2.FILLED)
            cv2.circle(img=self._img, center=chin, radius=5, color=(0, 255, 0), thickness=cv2.FILLED)

            # Draw lines between chinpoint and corners of the mouth 
            cv2.line(img=self._img, pt1=corner_left, pt2=chin, color=(0, 255, 0), thickness=1)
            cv2.line(img=self._img, pt1=corner_right, pt2=chin, color=(0, 255, 0), thickness=1)

    
    @property
    def img(self):
        return self._img
    
    @img.setter
    def img(self, value):
        print("setter of x called")
        self.load_img(value)


def main():
    analyzer = StaticAnalyzer()
    #analyzer.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/src/images/obama.jpg')
    analyzer.estimate_symmetry_line(draw=True)
    analyzer.quantify_eyebrows(draw=True)
    analyzer.quantify_mouth(draw=True)

    cv2.imshow('result', analyzer.img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()