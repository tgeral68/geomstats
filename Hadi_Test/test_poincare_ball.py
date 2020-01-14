import geomstats.backend as gs
from geomstats.geometry.poincare_ball import *
import torch

def poincare_ball_test():

    dimension = 3
    PoincareBall(dimension)
    Metric = PoincareMetric(dimension)

    #Our Distance
    x = gs.array([0.5, 0.5])
    y = gs.array([0.5, -0.5])
    d = Metric.dist(x,y)


    print('Distance Poincare', d)





if __name__ == "__main__":
    poincare_ball_test()


