import numpy as np
import matplotlib.pyplot as plt

def get_conf_region(region_bounds, granularity, model, n_a, n_b, C, L, F, Y, R):
    
    assert n_a + n_b == len(region_bounds)
    assert n_a + n_b == len(granularity)
    
    # Create a meshgrid for A and B values
    grid_axes = [np.linspace(region_bounds[i][0], region_bounds[i][1], granularity[i]) for i in range(n_a+n_b)]
    grid_vals = np.meshgrid(*grid_axes, indexing='ij')
    grid_vals = np.array(grid_vals)

    # Flatten the meshgrid and iterate through each (A,B) candidate
    flat_param_vals = [grid_vals[i].ravel() for i in range(n_a+n_b)]
    flat_param_vals = np.array(flat_param_vals).T
    pts_in_conf_region = []
    n_pts = flat_param_vals.shape[0]
    for i in range(n_pts):

        # Initialise this candidate's transfer functions
        point = flat_param_vals[i]
        A = np.concatenate(([1], -point[:n_a]))
        B = np.concatenate(([0],  point[n_a:]))
        G = (B, A)
        H = (C, A)

        # Transform to open loop
        try:
            G_0, H_0 = model.transform_to_open_loop(G, H, F, L)
        except Exception as e:
            # Unstable open loop system - ignore this infeasible parameter combination
            continue 

        # Check the SPS indicator and store the parameters if true
        in_sps, S1 = model.open_loop_sps(G_0, H_0, Y, R, n_a, n_b)  # Assuming Y and U are defined
        if in_sps:
            pts_in_conf_region.append(point)
    
    return np.array(pts_in_conf_region)

def plot_pts_in_conf_region(pts_in_conf_region, x_dim, y_dim, true_param, fig=None, ax=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots()
        ax.set_title('Points for which closed_loop_sps returns True')
    elif fig is None and ax is not None:
        raise Exception("fig,ax must both be specified")
    elif fig is not None and ax is None:
        raise Exception("fig,ax must both be specified")
    
    try:
        results = np.asnumpy(results)  # convert to np.ndarray
    except AttributeError:
        pass  # already an np.ndarray

    ax.plot(true_param[0], true_param[1], 'rx')  # Plot the true values as a red dot
    ax.plot(pts_in_conf_region[:, x_dim], pts_in_conf_region[:, y_dim], 'k.', markersize=3)
    
    return