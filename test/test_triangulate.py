import numpy as np
from math import sqrt
import random
from codim1.post.triangulate import *

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
    mesh, t_pts = triangulate(pts)
    changed = True
    i = 0
    while changed is True:
        mesh, t_pts, changed = refine_step(mesh, t_pts, lambda x: x > 0.5)
        i += 1
        if i > 3:
            changed = False
    plot(mesh, t_pts)
    plt.show()

def test_edge_length():
    points = [Point(0.0, 0.0, False, False),
              Point(1.0, 0.0, False, False),
              Point(0.0, 1.0, False, False),
              Point(1.0, 1.0, False, False)]
    t, t_pts = triangulate(points)
    edge_len = edge_lengths(t[0, :], t_pts)
    np.testing.assert_almost_equal(edge_len, [1.0, 1.0, sqrt(2)])

def test_refine():
    points = [Point(0.0, 0.0, False, False),
              Point(1.0, 0.0, False, False),
              Point(0.0, 0.01, False, False),
              Point(1.0, 0.01, False, False)]
    t, t_pts = triangulate(points)
    t, t_pts = refine_tri(t[0, :], t_pts, lambda x: x > 0.5)
    print t
    print t_pts
    np.testing.assert_almost_equal(t_pts[4, :], [0.5, 0.01])
    assert(t == [[3,4,0],[4,2,0]])

def test_refine():
    points = [Point(0.0, 0.0, False, False),
              Point(1.0, 0.0, False, False),
              Point(0.0, 0.01, False, False),
              Point(1.0, 0.01, False, False)]
    t, t_pts = triangulate(points)
    t, t_pts, changed = refine_step(t, t_pts, lambda x: x > 0.5)
    print t
    print t_pts

