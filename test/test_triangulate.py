import numpy as np
from math import sqrt
import random
from codim1.post.points_algorithm import *

def test_in_rect():
    p = Point(1,2,True, False)
    assert(in_rect(p, Rectangle(0, 2, 1, 3)) == True)
    assert(in_rect(p, Rectangle(-1, 0, 1, 3)) == False)

def test_clip_to_rect():
    keep_pt = Point(1,2, True, False)
    del_pt = Point(3,3, True, False)
    pts = [keep_pt, del_pt]
    new_pts = clip_to_rectangle(pts, Rectangle(0, 2, 1, 3))
    assert(new_pts[0] == keep_pt)
    assert(len(new_pts) == 1)

def random_points(count):
    pts = []
    for i in range(count):
        pts.append(Point(random.random(), random.random(), False, False))
    return pts

def test_plot():
    pts = random_points(40)
    plot(*triangulate(pts))

def test_edge_length():
    points = [Point(0.0, 0.0, False, False),
              Point(1.0, 0.0, False, False),
              Point(0.0, 1.0, False, False),
              Point(1.0, 1.0, False, False)]
    t, t_pts = triangulate(points)
    edge_len = edge_lengths(t_pts, t[0, :])
    np.testing.assert_almost_equal(edge_len, [1.0, 1.0, sqrt(2)])

def test_refine():
    points = [Point(0.0, 0.0, False, False),
              Point(1.0, 0.0, False, False),
              Point(0.0, 1.0, False, False),
              Point(1.0, 1.0, False, False)]
    t, t_pts = triangulate(points)
    t_pts, t = refine_tri(t_pts, t[0, :], lambda x: x > 0.1)
    np.testing.assert_almost_equal(t_pts[4, :], [0.5, 1.0])
    assert(t == [[3,4,0],[4,2,0]])

