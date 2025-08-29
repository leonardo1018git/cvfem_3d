import torch
from src.init_params import init, modeling


def solver(re_init, pho, inlet_velocity, viscosity, hydraulic_diameter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    points, point_nums, normalization_factor = modeling(hydraulic_diameter, device)
    points_numpy = points.cpu().numpy()

    u, v, w, p, delta_s, normal = init(point_nums, re_init, pho, device)



    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    w_numpy = w.cpu().numpy()
    p_numpy = p.cpu().numpy()
    delta_s_numpy = delta_s.cpu().numpy()
    normal_numpy = normal.cpu().numpy()

    return u, v, w, p
