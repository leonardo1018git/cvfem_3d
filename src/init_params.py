import torch


def modeling(hydraulic_diameter, device):
    # 网格点坐标
    points = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
        [1.0, 0.0, 3.0], [0.0, 1.0, 3.0], [-1.0, 0.0, 3.0], [0.0, -1.0, 3.0],
        [0.0, 0.0, 6.0], [1.0, 1.0, 6.0], [1.0, -1.0, 6.0], [-1.0, -1.0, 6.0], [-1.0, 1.0, 6.0]
    ], dtype=torch.float32, device=device) / hydraulic_diameter
    point_nums = (points.size())[0]

    # 网格
    mesh = torch.tensor

    x_length = torch.max(points[:, 0]) - torch.min(points[:, 0])
    y_length = torch.max(points[:, 1]) - torch.min(points[:, 1])
    z_length = torch.max(points[:, 2]) - torch.min(points[:, 2])
    normalization_factor = torch.max(torch.max(x_length, y_length), z_length).cpu().item()

    return points, point_nums, normalization_factor


def init(point_nums, re_init, pho, device):
    u = torch.zeros(point_nums, dtype=torch.float32, device=device)
    v = torch.zeros(point_nums, dtype=torch.float32, device=device)
    w = torch.zeros(point_nums, dtype=torch.float32, device=device)
    p = torch.zeros(point_nums, dtype=torch.float32, device=device)

    delta_s = torch.zeros(point_nums, dtype=torch.float32, device=device)
    normal = torch.zeros((point_nums, 3), dtype=torch.float32, device=device)


    return u, v, w, p, delta_s, normal
