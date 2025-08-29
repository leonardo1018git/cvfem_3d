from src.cvfem_solver import solver

if __name__ == "__main__":
    re_init = 1.0
    pho = 1.0
    inlet_velocity = 1.0
    viscosity = 1.0
    hydraulic_diameter = 2.0

    u, v, w, p = solver(re_init, pho, inlet_velocity, viscosity, hydraulic_diameter)
    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    w_numpy = w.cpu().numpy()
    p_numpy = p.cpu().numpy()

    print("done...")
