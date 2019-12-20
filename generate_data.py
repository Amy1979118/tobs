from dolfin import *
from dolfin_adjoint import *
from mshr import *
import numpy as np
import os
import convexhull


N = 100
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
mesh = Mesh(RectangleMesh(Point(0.0, 0.0), Point(delta, 1.0), int(N*3/2), N, diagonal="right"))
A = FunctionSpace(mesh, "DG", 0)        # control function space
pasta = "output_data/"
file_out = pasta + "data_out.csv"
if os.path.exists(file_out):
    os.remove(file_out)
file_data_in = File(pasta + "data_in.pvd")
file_new_mesh = File(pasta + "new_mesh.pvd")

class DistributionParallelLines(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edgePoints = []
        self.xa, self.ya = .8, .4
        self.xb, self.yb = .8/2, .4/2

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        if -(self.yb/self.xb)*x[0]+self.yb < x[1] < -(self.ya/self.xa)*x[0]+self.ya:
            values[0] = 1
            if near(x[0], 0, eps=1e-2) or \
                    near(x[0], 1.0, eps=1e-2) or \
                    near(x[1], 0.0, eps=1e-2) or \
                    near(x[1], 1.5, eps=1e-2):
                self.edgePoints.append((x[0], x[1]))

        if near(-(self.yb/self.xb)*x[0]+self.yb, x[1], eps=1e-2) or \
                near(-(self.ya/self.xa)*x[0]+self.ya, x[1], eps=1e-2):
            self.edgePoints.append((x[0], x[1]))

    def value_shape(self):
        return ()

def generate_parallel_lines():
    for xa in range(1, 15):
        xa /= 10
        for ya in range(1, 15):
            ya /= 10
            for ratio in [2, 3]:
                distrib = DistributionParallelLines()
                distrib.xa, distrib.ya = xa, ya
                distrib.xb, distrib.yb = xa/ratio, ya/ratio
                rho = interpolate(distrib, A)
                print("Os pontos sao: ({a}, {b}, r={c}".format(a=xa, b=ya, c=ratio))
                rho.rename("control", "")
                file_data_in << rho
                orderedPoints = convexhull.hull_pts(distrib.edgePoints)

                with open(file_out, "a+") as f:
                    f.write(str(orderedPoints))

                points = [Point(p[0], p[1]) for p in orderedPoints]

                shape = Polygon(points)
                new_mesh = generate_mesh(shape, 40)
                file_new_mesh << new_mesh


class DistributionCircles(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edgePoints = []
        self.xc, self.yc = .75, .5
        self.radius = 0.1

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        if (x[0]-self.xc)**2 + (x[1]-self.yc)**2 <= self.radius**2:
            values[0] = 1
            if near(x[0], 0, eps=1e-2) and \
                    near(x[1], 0.0, eps=1e-2) or \
                    near(x[0], 1.5, eps=1e-2) and \
                    near(x[1], 0.0, eps=1e-2) or \
                    near(x[0], 1.5, eps=1e-2) and \
                    near(x[1], 1.0, eps=1e-2) or \
                    near(x[0], 0.0, eps=1e-2) and \
                    near(x[1], 1.0, eps=1e-2):
                self.edgePoints.append((x[0], x[1]))
                self.edgePoints.append((x[0] + 1e-2, x[1] + 1e-2)) #to ensure the border will be there

        if near((x[0]-self.xc)**2 + (x[1]-self.yc)**2, self.radius**2, eps=1e-2):
            self.edgePoints.append((x[0], x[1]))

    def value_shape(self):
        return ()

def generate_circles():
    for xc in range(2, 15, 2):
        xc /= 10
        for yc in range(2, 15, 2):
            yc /= 10
            for radius in [4, 7, 9]:
                rad = radius/10
                distrib = DistributionCircles()
                distrib.xc, distrib.yc = xc, yc
                distrib.radius = rad
                rho = interpolate(distrib, A)
                print("Os pontos sao: ({a}, {b}, r={c}".format(a=xc, b=yc, c=rad))
                rho.rename("control", "")
                file_data_in << rho
                orderedPoints = convexhull.hull_pts(distrib.edgePoints)

                with open(file_out, "a+") as f:
                    f.write(str(orderedPoints))

                points = [Point(p[0], p[1]) for p in orderedPoints]

                points = points[0::2] #just getting half of points
                shape = Polygon(points)
                new_mesh = generate_mesh(shape, 40)
                file_new_mesh << new_mesh

class DistributionX(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edgePoints1 = []
        self.edgePoints2 = []
        self.xa, self.ya = .8, .4
        self.xb, self.yb = .8/2, .4/2

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        eps = 1e-2
        if -(self.yb/self.xb)*x[0]+self.yb < x[1] < -(self.ya/self.xa)*x[0]+self.ya:
            """ First line /
            """
            values[0] = 1

            if near(x[0], 0, eps) and \
                    near(x[1], 0.0, eps) or \
                    near(x[0], 1.5, eps) and \
                    near(x[1], 0.0, eps) or \
                    near(x[0], 1.5, eps) and \
                    near(x[1], 1.0, eps) or \
                    near(x[0], 0.0, eps) and \
                    near(x[1], 1.0, eps):
                self.edgePoints1.append((x[0], x[1]))

            if near(-(self.yb/self.xb)*x[0]+self.yb, x[1], eps) or \
                    near(-(self.ya/self.xa)*x[0]+self.ya, x[1], eps):
                self.edgePoints1.append((x[0], x[1]))

        if (self.yb/self.xb)*x[0] > x[1] > +(self.ya/self.xa)*x[0]-self.ya+self.yb:
            """ Second line
            """
            values[0] = 1

            if near(x[0], 0, eps) and \
                    near(x[1], 0.0, eps) or \
                    near(x[0], 1.5, eps) and \
                    near(x[1], 0.0, eps) or \
                    near(x[0], 1.5, eps) and \
                    near(x[1], 1.0, eps) or \
                    near(x[0], 0.0, eps) and \
                    near(x[1], 1.0, eps):
                self.edgePoints2.append((x[0], x[1]))

            if near(+(self.yb/self.xb)*x[0], x[1], eps) or \
                    near(+(self.ya/self.xa)*x[0]-self.ya+self.yb, x[1], eps):
                self.edgePoints2.append((x[0], x[1]))

    def value_shape(self):
        return ()

def generate_x():
    for xa in range(10, 16):
        xa /= 10
        for ya in range(10, 16):
            ya /= 10
            for ratio in [2, 3]:
                distrib = DistributionX()
                distrib.xa, distrib.ya = xa, ya
                distrib.xb, distrib.yb = xa/ratio, ya/ratio
                rho = interpolate(distrib, A)
                print("Os pontos sao: ({a}, {b}, r={c}".format(a=xa, b=ya, c=ratio))
                rho.rename("control", "")
                file_data_in << rho

                orderedPoints1 = convexhull.hull_pts(distrib.edgePoints1)
                with open(file_out, "a+") as f:
                    f.write(str(orderedPoints1))
                points = [Point(p[0], p[1]) for p in orderedPoints1]
                shape1 = Polygon(points)

                orderedPoints2 = convexhull.hull_pts(distrib.edgePoints2)
                with open(file_out, "a+") as f:
                    f.write(str(orderedPoints2))
                points = [Point(p[0], p[1]) for p in orderedPoints2]
                shape2 = Polygon(points)

                shape = shape1 + shape2

                new_mesh = generate_mesh(shape, 40)
                file_new_mesh << new_mesh

if __name__ == '__main__':
    # generate_parallel_lines()
    # generate_circles()
    generate_x()

