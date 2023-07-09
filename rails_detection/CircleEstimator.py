import numpy as np
from pyar import Point3D
from math import sin, cos, radians

class CircleEstimator:

    def __init__(self):
        pass

    def get_der(self, R, x0, y0):
        def der(x):
            if y0 <= -4.6:
                return -(x-x0)/((R*R - (x-x0)**2)**0.5)
            else:
                return +(x - x0) / ((R * R - (x - x0) ** 2) ** 0.5)
        return der

    def get_radius_center(self, surface_projections):
        data = []

        for pt in surface_projections:
            data.append((pt.x, pt.y))

        minimum = {}
        points = [[data[0], data[j], data[k]] for j in range(1, len(data) - 1) for k in range(j + 1, len(data))]
        for z in points:
            x1, y1 = z[0]
            x2, y2 = z[1]
            x3, y3 = z[2]
            try:
                y0 = (2 * (x1 - x3) * (y2 ** 2 - y1 ** 2 - x1 ** 2 + x2 ** 2) + 2 * (x1 - x2) * (
                        x1 ** 2 - x3 ** 2 - y3 ** 2 + y1 ** 2)) / (
                             4 * (y2 - y1) * (x1 - x3) - 4 * (x1 - x2) * (y3 - y1))
            except ZeroDivisionError:
                continue
            if x1 - x3 != 0:
                x0 = ((x1 ** 2 - x3 ** 2 - y3 ** 2 + y1 ** 2) + 2 * y0 * (y3 - y1)) / (2 * (x1 - x3))
            if x1 - x2 != 0:
                x0 = ((x1 ** 2 - x2 ** 2 - y2 ** 2 + y1 ** 2) + 2 * y0 * (y2 - y1)) / (2 * (x1 - x2))
            R = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** (1 / 2)

            n1, n2, n3 = 0, 0, 0

            for t in data:
                if round((t[0] - x0) ** 2 + (t[1] - y0) ** 2, 3) < round(R ** 2, 3):
                    n1 += 1
                elif round((t[0] - x0) ** 2 + (t[1] - y0) ** 2, 3) > round(R ** 2, 3):
                    n2 += 1
                else:
                    n3 += 1
            minimum[abs(n1 - n2)] = [
                '(x - {:.1f}) ** 2 + (y - {:.1f}) ** 2 = {:.1f} ** 2'.format(x0, y0, R).replace('- -', '+ '), z[0],
                z[1],
                z[2], n1, n2, n3, R, x0, y0]

        # print('Окружность:', minimum[min(minimum)][0])
        # print('Три точки, определяющие окружность:', str(minimum[min(minimum)][1]) + ',',
        # str(minimum[min(minimum)][2]) + ',', minimum[min(minimum)][3])
        # print('Радиус окружности:', minimum[min(minimum)][7])
        # print('Начальные координаты:', str(minimum[min(minimum)][8]) + ', ' + str(minimum[min(minimum)][9]))

        a0 = minimum[min(minimum)][8]
        b0 = minimum[min(minimum)][9]
        R = minimum[min(minimum)][7]

        return R, (a0, b0)

    def get_points(self, surface_projections):
        R, (a0, b0) = self.get_radius_center(surface_projections)

        points = []

        for i in range(360):
            new_i = radians(i)
            x = a0 + R * cos(new_i)
            y = b0 + R * sin(new_i)
            points.append((x, y))

        return points

    def get_error(self, surface_projections) -> float:
        circle = self.get_points(surface_projections)
        prs_circ = []
        for pt in circle:
            pt = (round(pt[0], 2), round(pt[1], 2), 0)
            prs_circ.append((pt[0], pt[1]))
        mins = []

        for pt in surface_projections:
            list_r = []
            for pt_c in prs_circ:
                list_r.append((((pt.x - pt_c[0]) ** 2) + (pt.y - pt_c[1]) ** 2) ** (1 / 2))
            mins.append(min(list_r) ** 2)
        return sum(mins)

    def getWheelAngle(self, surface_projections) -> float:
        R, (x0, y0) = self.get_radius_center(surface_projections)
        derivative = self.get_der(R, x0, y0)
        tan = derivative(0)
        if tan >= 0:
            wheel_angle = np.pi / 2 - np.arctan(tan)
        else:
            wheel_angle = -np.pi / 2 - np.arctan(tan)
        return wheel_angle
