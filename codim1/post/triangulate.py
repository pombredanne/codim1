import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import namedtuple

Point = namedtuple('Point', 'x, y, is_boundary, refinement_pt')
Rectangle = namedtuple('Rectangle', 'x_l, x_r, y_d, y_u')

def in_rect(pt, rect):
    if rect.x_l < pt.x < rect.x_r and\
       rect.y_d < pt.y < rect.y_u:
        return True
    return False

def clip_to_rectangle(points, rectangle):
    new_points = [p for p in points if in_rect(p, rectangle)]
    return new_points

def plot(tris, pts_numpy):
    plt.triplot(pts_numpy[:, 0], pts_numpy[:, 1], tris)
    plt.plot(pts_numpy[:,0], pts_numpy[:,1], 'o')

def triangulate(points):
    pts_list = [(p.x, p.y) for p in points]
    pts_numpy = np.array(pts_list)
    triang = Delaunay(pts_numpy)
    return triang.simplices, pts_numpy

edges = [[1, 0], [2, 1], [0, 2]]

def edge_lengths(points, triangle):
    return [np.linalg.norm(points[triangle[e[0]], :] -
                           points[triangle[e[1]], :])
            for e in edges]

def refine_tri(points, triangle, criteria):
    edge_len = edge_lengths(points, triangle)
    new_points = points
    new_triangles = triangle
    for i, e in enumerate(edges):
        if criteria(edge_len):
            new_pt = (points[triangle[e[0]], :] + \
                      points[triangle[e[1]], :]) / 2.0
            new_idx = new_points.shape[0]
            new_points = np.vstack((new_points, new_pt))
            new_triangles = [[triangle[0], triangle[1], triangle[2]],
                             [triangle[0], triangle[1], triangle[2]]]
            new_triangles[0][e[0]] = new_idx
            new_triangles[1][e[1]] = new_idx
            break
    return new_points, new_triangles

#use scipy kdtree for fast near neighbors
