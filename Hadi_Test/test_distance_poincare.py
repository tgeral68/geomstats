

import torch
import numpy as np
from geomstats.geometry.hyperbolic_space import HyperbolicSpace as hs
import geomstats.backend as gs

def add(x, y):
    nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x)
    ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x)
    xy = (x * y).sum(-1, keepdim=True).expand_as(x)
    return ((1 + 2*xy+ ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)

def log(k, x):
    kpx = add(-k,x)
    norm_kpx = kpx.norm(2,-1, keepdim=True).expand_as(kpx)
    norm_k = k.norm(2,-1, keepdim=True).expand_as(kpx)
    res = (1-norm_k**2)* ((torch.arc_tanh(norm_kpx))) * (kpx/norm_kpx)
    if(0 != len((norm_kpx==0).nonzero())):
        res[norm_kpx == 0] = 0
    return res

class PoincareDistance2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        with torch.no_grad():
            # eps = torch.randu(x.shape[-1], device=x.device)
            x_norm = torch.clamp(torch.sum(x ** 2, dim=-1), 0, 1-1e-3)
            y_norm = torch.clamp(torch.sum(y ** 2, dim=-1), 0, 1-1e-3)
            d_norm = torch.sum((x-y) ** 2, dim=-1)
            cc = 1+2*d_norm/((1-x_norm)*(1-y_norm))
            dist = torch.log(cc + torch.sqrt(cc**2-1))
            ctx.save_for_backward( x, y, dist)
            return  dist
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            x, y, dist = ctx.saved_tensors
            res_x, res_y =  (- (log(x, y)/(dist.unsqueeze(-1).expand_as(x))) * grad_output.unsqueeze(-1).expand_as(x),
                     - (log(y, x)/(dist.unsqueeze(-1).expand_as(x))) * grad_output.unsqueeze(-1).expand_as(x))
            # print(res_y)
            if((dist == 0).sum() != 0):
                # it exist example having same representation
                res_x[dist == 0 ] = 0
                res_y[dist == 0 ] = 0
            return res_x, res_y


def poincare_distance(x, y):
    return PoincareDistance2.apply(x, y)

if __name__ == "__main__":


    #Our Distance
    x = torch.from_numpy(np.array([0.5, 0.5]))
    y = torch.from_numpy(np.array([0.5, -0.5]))
    d = poincare_distance(x,y)



    #GeomStat Distance
    dimension = 2
    # metrics = hm(dimension)
    space = hs(dimension)



    METRIC = space.metric

    point_a = gs.array([35, -25, -25])


    point_b = gs.array([35, -25, 25])



    a_intrinsic = space.extrinsic_to_intrinsic_coords(point_a)
    b_intrinsic = space.extrinsic_to_intrinsic_coords(point_b)

    c_intrinsic = gs.array([0.5,0.5])
    d_intrinsic = gs.array([0.5,-0.5])

    e_intrinsic = gs.array([-0.5,0.5])
    f_intrinsic = gs.array([-0.5,-0.5])


    c_extrinsic = space.intrinsic_to_extrinsic_coords(c_intrinsic)
    d_extrinsic = space.intrinsic_to_extrinsic_coords(d_intrinsic)
    e_extrinsic = space.intrinsic_to_extrinsic_coords(e_intrinsic)
    f_extrinsic = space.intrinsic_to_extrinsic_coords(f_intrinsic)

    belong_a = space.belongs(point_a)
    belong_b = space.belongs(point_b)
    belong_c = space.belongs(c_extrinsic)
    belong_d = space.belongs(d_extrinsic)

    res = METRIC.dist(point_a, point_b)

    a_our = torch.from_numpy(a_intrinsic)
    b_our = torch.from_numpy(b_intrinsic)
    c_our = torch.from_numpy(c_intrinsic)
    d_our = torch.from_numpy(d_intrinsic)
    e_our = torch.from_numpy(e_intrinsic)
    f_our = torch.from_numpy(f_intrinsic)

    distance_c_d_geom = METRIC.dist(c_extrinsic,d_extrinsic)
    distance_c_d_our = poincare_distance(c_our, d_our)

    distance_e_f_geom = METRIC.dist(e_extrinsic,f_extrinsic)
    distance_e_f_our = poincare_distance(e_our, f_our)

    distance_c_e_geom = METRIC.dist(c_extrinsic, e_extrinsic)
    distance_c_e_our = poincare_distance(c_our, e_our)

    distance_d_f_geom = METRIC.dist(d_extrinsic, f_extrinsic)
    distance_d_f_our = poincare_distance(d_our,f_our)

    print("Distance geomstats ", distance_c_d_geom)
    print('Distance our', distance_c_d_our)



    print("Distance geomstats 1 ", distance_e_f_geom)
    print('Distance our 1', distance_e_f_our)

    print("Distance geomstats ", distance_c_e_geom)
    print('Distance our', distance_c_e_our)

    print("Distance geomstats 1 ", distance_d_f_geom)
    print('Distance our 1', distance_d_f_our)


    print("Point a extrinsic", point_a)
    print("Point a intrinsic", a_intrinsic)
    print("Point a belongs", belong_a)

    print("Point b extrinsic", point_b)
    print("Point b intrinsic", b_intrinsic)
    print("Point b belongs", belong_b)

    print("Point c intrinsic", c_intrinsic)
    print("Point c extrinsic", c_extrinsic)
    print("Point c belongs", belong_c)

