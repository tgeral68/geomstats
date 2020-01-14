import geomstats.backend as gs
from geomstats import visualization
from geomstats.geometry.hyperbolic_space import HyperbolicMetric as hm
from geomstats.geometry.hyperbolic_space import HyperbolicSpace as hs
from geomstats.visualization import *
import matplotlib.pyplot as plt
import torch

SQUARE_SIZE = 50






def poincare_ball_test():

    dimension = 2
    #metrics = hm(dimension)
    space = hs(dimension)

    METRIC= space.metric

    point_a = gs.array([35, -25, -25])
    point_b = gs.array([35,-25,25])

    res = METRIC.dist(point_a, point_b)



    print("Distance ", res)




    top = SQUARE_SIZE / 2.0
    bot = - SQUARE_SIZE / 2.0
    left = - SQUARE_SIZE / 2.0
    right = SQUARE_SIZE / 2.0
    corners_int = [(bot, left), (bot, right), (top, right), (top, left)]
    corners_ext = H2.intrinsic_to_extrinsic_coords(corners_int)

    print("corners_ext", corners_ext)

    n_steps = 20
    ax = plt.gca()



    for i, src in enumerate(corners_ext):

        dst_id = (i+1) % len(corners_ext)
        dst = corners_ext[dst_id]
        tangent_vec = METRIC.log(point=dst, base_point=src)
        geodesic = METRIC.geodesic(initial_point=src,
                                   initial_tangent_vec=tangent_vec)
        t = np.linspace(0, 1, n_steps)


        edge_points = geodesic(t)


        gs.append(edge_points,point_a)
        gs.append(edge_points,point_b)

        visualization.plot(
            edge_points,
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color='black')




    print('Plot added points',)

    print('Plot points', edge_points)

    plt.show()

    #Visualization test following the Poincare_Disk example
    #Use the class that is inside the visualization.py



    # point_c = gs.array([0.5,0.5, gs.sqrt(5)])
    # point_d = gs.array([-0.5,-0.5, gs.sqrt(5)])
    # D = PoincareDisk(points = [point_c,point_d])
    # D.convert_to_poincare_coordinates()
    # D.draw()
    # point_a = gs.array([0.5, 0.5, 0.2])
    # point_b = gs.array([0.7, 0.2, 0.3])
    # res = metrics.dist(point_a, point_b)
    # print("Distance ", res)

if __name__ == "__main__":
    poincare_ball_test()

