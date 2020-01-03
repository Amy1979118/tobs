from dolfin import *
from dolfin_adjoint import *
from mshr import *
import numpy as np
import os
import convexhull

from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator
import matplotlib.pyplot as plt
from polylidar import extractPlanesAndPolygons
from polylidarutil import (generate_test_points, plot_points, plot_triangles, get_estimated_lmax,
                            plot_triangle_meshes, get_triangles_from_he, get_plane_triangles, plot_polygons)
import shapely.geometry as shape


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
    for xa in range(10, 11):
        xa /= 10
        for ya in range(10, 11):
            ya /= 10
            for ratio in [2]:
                distrib = DistributionX()
                distrib.xa, distrib.ya = xa, ya
                distrib.xb, distrib.yb = xa/ratio, ya/ratio
                rho = interpolate(distrib, A)
                print("Os pontos sao: ({a}, {b}, r={c}".format(a=xa, b=ya, c=ratio))
                rho.rename("control", "")
                file_data_in << rho
                return rho


def generate_point_list(rho, mesh):
    point_cloud = []
    for celula in cells(mesh):
        xy = celula.get_vertex_coordinates()
        xg = (xy[0] + xy[2] + xy[4])/3.
        yg = (xy[1] + xy[3] + xy[5])/3.
        if rho(xg,yg) > 0.6:
            point_cloud.append([xg, yg])
    return np.array(point_cloud)

def get_point(p_index, points):
    if points.shape[1] > 2:
        return [points[p_index, 0], points[p_index, 1], points[p_index, 2]]
    else:
        return [points[p_index, 0], points[p_index, 1]]

def generate_polygon(rho, mesh, geo='2D', accept_holes=False):
    point_cloud = generate_point_list(rho, mesh)
    if geo == '2D':
        delaunay, planes, polygons = extractPlanesAndPolygons(point_cloud, xyThresh=0.0, alpha=0.0, lmax=0.05, minTriangles=5)
        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
        plot_points(point_cloud, ax)
        plot_polygons(polygons, delaunay, point_cloud, ax)
        for poly in polygons:
            shell_coords = [get_point(p_index, point_cloud) for p_index in poly.shell]
            try:
                poly.holes[0]
            except:
                accept_holes=False
            if accept_holes:
                hole_coords = [get_point(p_index, point_cloud) for p_index in poly.holes[0]]
        # shape = shape.Polygon([ [item[0], item[1]] for item in shell_coords])
        # hole_coords = hole_coords[0::2]
        if accept_holes:
            shape_hole = Polygon([Point(item[0], item[1]) for item in hole_coords])

        shell_coords = shell_coords[0::2]
        shell_coords.reverse()
        shape = Polygon([Point(item[0], item[1]) for item in shell_coords])

        if accept_holes: new_mesh = generate_mesh(shape - shape_hole, 20)
        else: new_mesh = generate_mesh(shape, 20)

    else:
        raise NotImplementedError()

    return new_mesh

def generate_polygon_refined(rho, mesh, geo="2D", accept_holes=False):
    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    new_mesh = generate_polygon(rho, mesh, geo, accept_holes)
    regions = MeshFunction("bool", new_mesh, False)
    regions.set_all(0)
    region_to_refine = Border()
    region_to_refine.mark(regions, True)

    new_mesh_refined = refine(new_mesh, regions)

    domain = MeshFunction("size_t", new_mesh, 1)
    domain.set_all(0)
    edge = Border()
    edge.mark(domain, 1)

    return new_mesh_refined, domain


if __name__ == '__main__':
    pasta = "output_smooth/"
    file_out = pasta + "data_out.csv"
    if os.path.exists(file_out):
        os.remove(file_out)
    file_data_in = File(pasta + "data_in.pvd")
    file_new_mesh = File(pasta + "new_mesh.pvd")

    print("***This module was executed directed as a demo execution***")
    rho = generate_x()
    raise Exception("Fix the mesh entries")

    new_mesh_refined, domain = generate_polygon_refined(rho, mesh)
    File(pasta + "new_mesh_refined.pvd") << new_mesh_refined
    File(pasta + "regions.pvd") << domain

