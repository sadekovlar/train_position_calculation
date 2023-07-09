import cv2
import numpy as np
from srccam import CalibReader, Calib, Camera, Point3d
from pyar import Camera as pyarCamera, Size
from pyar import Point2D
from pyar import Point3D
import matplotlib.pyplot as plt
from statistics import mean
from math import sin, cos, radians

from rails_detection.CircleEstimator import CircleEstimator
from rails_detection.PolynomeEstimator import PolynomeEstimator

class Interpolate():
    def __init__(self, path):
        file_name = path + '/calib/leftImage.yml'
        self.pyar_camera = pyarCamera.from_yaml(file_name)

    def get_line_projection_points(self, image) -> list[Point2D]:
        #hsv_min = (0, 200, 200)
        #hsv_max = (50, 255, 255)
        #kernel = np.ones((2, 2), 'uint8')

        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        # thresh = cv2.erode(thresh, kernel, iterations=1)
        #cv2.imshow('thresh', image)

        non_zero_coordinates = cv2.findNonZero(image)
        result = [Point2D(pt[0]) for pt in non_zero_coordinates]
        return result

    def get_surface_projections(self, pyar_camera, points) -> list[Point2D]:
        wheel = (0, -4.6, 0)
        points_3d = []
        for point in points:
            point_3d = pyar_camera.reproject_point_with_height(point, 0)
            points_3d.append(point_3d)
    
        points_3d = [(pt.x, pt.y, pt.z) for pt in points_3d]
    
        points_3d = [wheel] + points_3d
        points_on_surface = [(pt[0], pt[1]) for pt in points_3d]
        points_on_surface.sort()
        points_on_surface = [Point2D(pt) for pt in points_on_surface]
    
        return points_on_surface

    def draw_points(self, image, pyar_camera, points, color):
        polyline_3d = np.array([(pt[0], pt[1], 0) for pt in points])
        for pt_3d in polyline_3d:
            pt_2d = pyar_camera.project_point(pt_3d)
            pt = (int(pt_2d.x), int(pt_2d.y))
            cv2.circle(image, pt, 2, color, 2)

    def chooseEstimator(self, surface_projections):
        circle_error = CircleEstimator().get_error(surface_projections)
        polynome_error = PolynomeEstimator().get_error(surface_projections)

        hypotheses = {
            circle_error: CircleEstimator(),
            polynome_error: PolynomeEstimator()
        }
        least_error = min(hypotheses.keys())

        print("Ошибка на окружности: ", circle_error)
        print("Ошибка на полиномах: ", polynome_error)
        return hypotheses[least_error]


'''
        for pt in circle:
            pt = (round(pt[0], 2), round(pt[1], 2), 0)
            if (abs(pt[0]) == 0 and round(pt[1], 1) == -4.6) or pt[1] >= 0:
                prs_circ.append((pt[0], pt[1]))
                #pr_pyar = self.pyar_camera.project_point(Point3D(pt))
                    #pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
                    #cv2.circle(image, pr, 4, (0, 0, 255), 3)

            #cv2.imshow('image', image)
    '''
