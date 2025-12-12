import numpy as np
import time
from dolfin import Point
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from dolfin import SpatialCoordinate
import numpy as np

def sensor_output(u, sensor_loc):
    mesh = u.function_space().mesh()
    vertices = u.compute_vertex_values(mesh)

    values = []
    for vid in sensor_loc:
        values.append(vertices[vid])
    return np.array(values)


def plot_field(u, ax=None, vmin=0.0, vmax=1.9):
    mesh = u.function_space().mesh()
    coords = mesh.coordinates()
    x = coords[:,0]; y = coords[:,1]
    cells = mesh.cells()
    triang = tri.Triangulation(x, y, cells)
    values = u.compute_vertex_values(mesh)

    levels = np.linspace(vmin, vmax, 100)
    norm = plt.Normalize(vmin, vmax)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    # plot field
    tcf = ax.tricontourf(triang, values, levels=levels, cmap='inferno', norm=norm)
    ax.set_aspect("equal")
    fig.tight_layout()

    # return plot objects
    return fig, ax, tcf


def plot_sensor(mesh, vid):
    coords = mesh.coordinates()
    x = coords[:,0]
    y = coords[:,1]
    cells = mesh.cells()

    tri_mesh = tri.Triangulation(x, y, cells)

    fig, ax = plt.subplots(figsize=(6,3))

    # plot mesh
    ax.triplot(tri_mesh, color='k', linewidth=0.3)

    # plot sensors
    for id in vid:
        vx, vy = coords[id]
        ax.scatter(vx, vy, s=5, marker='s',facecolor='red')

        
def cost_fun(problem, sensor_loc, observation, mu0, rb=False):
    '''
    Objective function for inverse identification of q

    Inputs:
    problem: reduced problem if rb=True (or truth problem if rb=False)
    sensor loc: list of sensor locations [(x1,y1), (x2,y2), ...]
    observation: sensor measurements (true)
    mu0: initial guess of q
    rb: True if simulating with RB model
    '''

    mu0 = list(mu0)

    def cost(theta): # objective function
        theta = float(theta)
        mu = mu0.copy()
        mu[2] = theta # parameter to optimize (q)

        problem.set_mu(tuple(mu))

        if rb:
            u = problem.solve() # RB temperature field
            u = problem.basis_functions * u # RB temperature field

        else:
            reset_cache(problem)
            u = problem.solve() # FE temperature field

        yout = sensor_output(u, sensor_loc) # simulation output

        err = yout - observation
        return 0.5*float(np.dot(err,err))
    return cost


def cost_fun_with_error(problem, sensor_loc, observation, mu0,
                        kriging):
    
    def cost(theta):

        theta = float(theta)
        mu = mu0.copy()
        mu[2] = theta

        # run RB
        problem.set_mu(tuple(mu))
        u_rb_coeff = problem.solve()
        u_rb = problem.basis_functions * u_rb_coeff
        y_rb = sensor_output(u_rb, sensor_loc)

        # predict RB error from Kriging
        mu_arr = np.array(mu).reshape(1,-1)
        err_pred = kriging.predict(mu_arr)
        err_pred = float(err_pred.ravel()[0])

        # get corrected RB output
        y_correct = y_rb + err_pred

        err = observation - y_correct

        return 0.5 * float(np.dot(err, err))

    return cost


def identify(problem, sensor_loc, y_true, online_mu,
               initial_guess=1.0, rb=False):

    '''
    Inverse identification of parameter q based on sensor measurements 
    using RB (or FE) model

    Inputs:
        problem: reduced problem if rb=True (or truth problem if rb=False)
        sensor loc: list of vertex indices where sensors are located
        y_true: sensor measurements (true)
        online_mu: known parameters and initial guess of q
        initial_guess: initial guess of q
        rb: True if simulating with RB model

    Outputs:
        q_opt: estimated parameter q
    '''

    # set mu
    mu0 = list(online_mu)
    mu0[2] = initial_guess

    # define cost function
    cost = cost_fun(problem, sensor_loc, y_true, mu0, rb=rb)
    theta0 = mu0[2]

    start_time = time.time()
    res = minimize(
        cost,
        x0=theta0,
        method='BFGS',
        options={'disp': False, 'eps': 1e-8}
    )
    end_time = time.time()
    print("Identification time:", end_time - start_time)
    q_opt = float(res.x)

    new_mu = mu0.copy()
    new_mu[2] = q_opt

    return new_mu


def identify_with_error(
    problem, sensor_loc, y_true, online_mu,
    kriging,
    initial_guess=1.0):

    '''
    Inverse identification of parameter q based on sensor measurements 
    using RB model. The identification takes into account RB model error.

    Inputs:
        problem: reduced problem
        sensor_loc: list of vertex indices where sensors are located
        y_true: sensor measurements (true)
        online_mu: known parameters and initial guess of q
        kriging: trained Kriging model to predict RB error
        initial_guess: initial guess of q
    '''

    # set mu
    mu0 = list(online_mu)
    mu0[2] = initial_guess

    # define cost function
    cost = cost_fun_with_error(problem, sensor_loc, y_true, mu0, kriging=kriging)
    theta0 = mu0[2]

    start_time = time.time()
    res = minimize(
        cost,
        x0=theta0,
        method='BFGS',
        options={'disp': False, 'eps': 1e-8}
    )
    end_time = time.time()
    print("Identification time using error corrected RB model:", end_time - start_time)

    q_opt = float(res.x)

    new_mu = mu0.copy()
    new_mu[2] = q_opt

    return new_mu


def reset_cache(problem):
    class DummyCache(dict):
        def __getitem__(self, key): raise KeyError
        def __setitem__(self, key, value): pass
        def clear(self): pass

    problem._solution_cache = DummyCache()
    problem._latest_solve_kwargs = {}

    if hasattr(problem, "_is_solving"):
        del problem._is_solving


def read_sensor(truth_problem, mu, q_value, sensor_loc):
    # update mu
    mu_new = list(mu)
    mu_new[2] = q_value

    # run FE simulation to get the true measurement
    reset_cache(truth_problem)
    truth_problem.set_mu(tuple(mu_new))
    truth_solution = truth_problem.solve()

    # get sensor output
    y_true = sensor_output(truth_solution, sensor_loc)

    return y_true, truth_solution


def monitor(truth_problem, mu):

    vmin, vmax = 0.0, 1.9
    cmap='inferno' # set colorbar

    # update mu
    est_mu = list(mu)
    print('system parameters for monitoring:', est_mu)

    # run FE simulation
    truth_problem.set_mu(tuple(est_mu))
    start_time = time.time()
    truth_solution = truth_problem.solve()
    end_time = time.time()

    # plot
    fig, ax, tcf = plot_field(truth_solution, vmin=0.0, vmax=1.9)
    cbar = fig.colorbar(tcf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Temperature")
    cbar.ax.tick_params(labelsize=8)
    plt.show()

    # print maximum temperature
    max_temp = truth_solution.vector().max()
    print("current maximum temperature:", max_temp)

    return truth_solution, max_temp

def control_cooling(truth_problem, current_mu, max_temp,
                    low_threshold=0.7, high_threshold=1.5):

    vmin, vmax = 0.0, 1.9
    cmap='inferno'

    # --------------------------
    # Scenario 1: When temperature is already low -> stop cooling
    # --------------------------
    if max_temp < low_threshold:
        print("Temperature is already low. Reducing cooling.")
        new_mu = list(current_mu)
        new_mu[0] = 2   # your chosen control action
        new_mu[1] = 1
        print("system parameters after control:", new_mu)

        # run FE simulation
        truth_problem.set_mu(tuple(new_mu))
        start_time = time.time()
        truth_solution = truth_problem.solve()
        end_time = time.time()

        # plot field
        fig, ax, tcf = plot_field(truth_solution, vmin=0.0, vmax=1.9)
        cbar = fig.colorbar(tcf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Temperature")
        cbar.ax.tick_params(labelsize=8)
        plt.show()

        # print maximum temperature
        new_max_temp = truth_solution.vector().max()
        print("current maximum temperature:", new_max_temp)

        return new_mu, truth_solution, new_max_temp

    # --------------------------
    # Scenario 2: When tempearture is too high -> increase cooling
    # --------------------------
    if max_temp > high_threshold:
        print("Temperature is too high. Increasing cooling.")
        new_mu = list(current_mu)
        new_mu[0] = 5
        new_mu[1] = 3
        print("system parameters after control:", new_mu)

        truth_problem.set_mu(tuple(new_mu))
        start_time = time.time()
        truth_solution = truth_problem.solve()
        end_time = time.time()
        
        fig, ax, tcf = plot_field(truth_solution, vmin=0.0, vmax=1.9)
        cbar = fig.colorbar(tcf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Temperature")
        cbar.ax.tick_params(labelsize=8)
        plt.show()

        new_max_temp = truth_solution.vector().max()
        print("current maximum temperature:", new_max_temp)

        return new_mu, truth_solution, new_max_temp

    # --------------------------
    # If temperature is in the safe range -> do nothing
    # --------------------------
    print("Temperature is in normal range. No cooling adjustment.")
    truth_solution = truth_problem.solve()
    new_max_temp = truth_solution.vector().max()

    return current_mu, truth_solution, new_max_temp
