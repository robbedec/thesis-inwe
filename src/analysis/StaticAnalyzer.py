import cv2
import sys
from enum import Enum

sys.path.append('../keypoints/detectors')
sys.path.append('../utils')
from MediapipeKPDetector import MediapipeKPDetector
from util import dist_point_to_line, dist_point_to_point, mean_position, round_tuple

class Measurements(Enum):
    Eyebrows = 1

class StaticAnalyzer():

    def __init__(self):
        """
        Throws an assertion error if no faces were detected 
        """

        self._detector = MediapipeKPDetector(maxFaces=1)
        #self.img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/notebooks/images/obama.jpg')
        self._img = cv2.imread('/home/robbedec/repos/ugent/thesis-inwe/data/MEEI_Standard_Set/Flaccid/CompleteFlaccid/CompleteFlaccid1/CompleteFlaccid1_1.jpg')

        # Detect facial landmarks
        # Passing a copy to the detector prevents default drawings by mediapipe
        self._img_mp, self._faces = self._detector.findFaceMesh(self._img.copy(), draw_points=True, draw_indices=False, filtered=False)
        assert len(self._faces) > 0

        self._face = self._faces[0]

    def estimate_symmetry_line(self, draw=True):
        """
        Finds the equation of the line that connects the two inner corners of
        the eyes and calculates the perpendicular line that passes through the
        center of the first line 

        Returns: The slope of the perpendicular line together with a point that lies on it.
        """
        print(dist_point_to_line(5, (0, 8), (1,1)))

        img_x, img_y, _ = self._img.shape

        # Draw line through the inside eye corners (points 133 & 362 for unfiltered keypoints or 39 & 42)
        (x1, y1) = self._faces[0][133]
        (x2, y2) = self._faces[0][362]

        # Point 168 is halfway between the eyes
        x_mid, y_mid = ((x1 + x2) / 2, (y1 + y2) / 2)

        horizontal_slope = (y2 - y1) / (x2 - x1)
        vertical_slope = x_mid if horizontal_slope == 0 else -1 / horizontal_slope


        if draw:
            # Form: y=f(x)
            horizontal_line = lambda x: (horizontal_slope) * (x - x1) + y1

            # Form x=f(y), be careful when horizontal slope = 0
            vertical_line = lambda y: ((y - y_mid) / vertical_slope) + x_mid

            cv2.line(self._img, (0, round(horizontal_line(0))), (img_x, round(horizontal_line(img_x))), (0, 255, 0), thickness=1)
            cv2.line(self._img, (round(vertical_line(0)), 0), (round(vertical_line(img_y)), img_y), (0, 255, 0), thickness=1)

        return vertical_slope, (x_mid, y_mid)

    def quantify_eyebrows(self):
        # https://pdfs.semanticscholar.org/b436/2cd87ad219790800127ddd366cc465606a78.pdf

        # Find mean coördinate in left eyebrow
        MLEB = mean_position([70, 63, 105, 66, 107], self._face)
        print(MLEB)
        
        # Find mean coördinate in right eyebrow
        MREB = mean_position([336, 296, 334, 293, 300], self._face)
        print(MREB)

        # Mean positions of the eyes (iris)
        # Approximates the center fo the eye
        LEC = mean_position([473, 474, 475, 476, 477], self._face)
        REC = mean_position([468, 469, 470, 471, 472], self._face)

        D_EB_EYE_L = dist_point_to_point(MLEB, LEC)
        D_EB_EYE_R = dist_point_to_point(MREB, REC)

        print(D_EB_EYE_L / D_EB_EYE_R)

        self._img = cv2.circle(self._img, round_tuple(LEC), 5, (0, 255, 0), cv2.FILLED)
        self._img = cv2.circle(self._img, round_tuple(REC), 5, (0, 255, 0), cv2.FILLED)
        self._img = cv2.circle(self._img, round_tuple(MLEB), 5, (0, 255, 0), cv2.FILLED)
        self._img = cv2.circle(self._img, round_tuple(MREB), 5, (0, 255, 0), cv2.FILLED)

    
    @property
    def img(self):
        return self._img
    
    @img.setter
    def img(self, value):
        print("setter of x called")
        self._img = value


def main():
    analyzer = StaticAnalyzer()
    #analyzer.estimate_symmetry_line()
    analyzer.quantify_eyebrows()

    cv2.imshow('result', analyzer.img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()