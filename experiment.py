import matplotlib.pyplot as plt
import numpy as np
from Train import Train
from rails_detection.interpolation import Interpolate
import cv2
import os
from pyar import Point2D
from pyar import Point3D
from rails_detection.CircleEstimator import CircleEstimator
from rails_detection.PolynomeEstimator import PolynomeEstimator

def drawSurfToImage(camera, surface_projections):
    prs = []
    for pt in surface_projections:
        pt = (pt.x, pt.y, 0)
        pr_pyar = camera.project_point(Point3D(pt))
        pr = tuple(map(int, [pr_pyar.x, pr_pyar.y]))
        prs.append(pr)
        cv2.circle(draw_image, pr, 5, (255, 250, 20), 2)
    cv2.imshow('draw_image', draw_image)
    cv2.waitKey(3)

def drawSurf(surface_projections):
    point_x = []
    point_y = []
    for point in surface_projections:
        point_x.append(point.x)
        point_y.append(point.y)
    plt.plot(np.array(point_x), np.array(point_y),"r.")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

path = "../get.163"
path = os.path.abspath(path)
files = os.listdir(path)
files = [file.replace(".segm.png","") for file in files if file.find("segm.png") > 0]
numbers = np.array(files, dtype = 'int')
numbers = np.sort(numbers)

# добавил
inter = Interpolate(path)  
for number in numbers:
    segm_im_path = os.path.join(path, f"{number}.segm.png")
    im_path = os.path.join(path, f"{number}.png")
    segm_img = cv2.imread(segm_im_path)
    segm_img = cv2.cvtColor(segm_img, cv2.COLOR_BGR2GRAY)
    pixels = segm_img == 1
    new_image = np.zeros_like(segm_img)
    # выделяем дорогу другим цветом
    new_image[pixels] = 255
    # тут его можно использовать
    draw_image = np.zeros_like(segm_img)
    points = inter.get_line_projection_points(new_image)
    surface_projections = inter.get_surface_projections(inter.pyar_camera, points)
    #drawSurfToImage(inter.pyar_camera, surface_projections)
    #drawSurf(surface_projections)
    # вычисляем оптимальную кривую которая лежит в данной области
    estimator = inter.chooseEstimator(surface_projections[:100])
    wheel_angle = estimator.getWheelAngle(surface_projections) / np.pi * 180
    print("Угол поворота передней тележки: ", wheel_angle, "градусов")
    plt.axis('equal')
    X = [pt.x for pt in surface_projections]
    Y = [pt.y for pt in surface_projections]
    plt.scatter(X, Y)
    all_circle_points = CircleEstimator().get_points(surface_projections)
    all_poly_points = PolynomeEstimator().get_points(surface_projections)
    c_points = []
    p_points = []
    for pt in all_circle_points:
        if -30 <= pt[0] <= 30:
            c_points.append(pt)
    x = [pt[0] for pt in c_points]
    y = [pt[1] for pt in c_points]
    plt.plot(x, y)
    plt.show()
    inter.draw_points(image, pyar_camera, all_circle_points, (200, 100, 100))
    #draw_points(image, pyar_camera, all_poly_points, (100, 200, 100))
    cv2.putText(image, "ANGLE: "+str(wheel_angle), (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow('result', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("==============================")

#img = cv2.imread(im_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image", new_image)
    #cv2.waitKey(2)
cv2.destroyAllWindows()
