"""
ENME403 Pendulum Lab 2019S1
Author: Campbell McDiarmid

1. Prove that the Open Loop system is unstable
2. Model all 4 configurations and:
    a. Select 1 set of poles and determine the gains (pole placement)
    b. Create 1 set of gains using LQR optimal control theory.
3. From the resulting 8 sets if gains, select 1 pole placement set and 1 LQR set of gains that are robust enough to
   control all 4 possible combinations
4. Bring these 2 sets of gains, on paper, to the lab class

Extra: prize for least controller effort and greatest sensor noise tolerance

Design criteria:
* r(t) = 0.1m
* ts for x and theta < 5-7s
* tr for x < 1-3s
* Mp < 10 degrees
* ess < 2%
* V < 10V
* dV/dt < 30V/s
* (Mass and other values noted will likely differ to the real physical values)
"""

########################################################################################################################
#
#                                                    IMPORTS
#
########################################################################################################################


from numpy import array, zeros, ones, pi, exp, diag, linspace, diff
from numpy.linalg import inv
from numpy.linalg import eigvals as eig
from scipy.signal import tf2ss, ss2tf, step, impulse, place_poles, lsim
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
#
#                                                    CONSTANTS
#
########################################################################################################################


# Value ranges for +0g, +200g, +300g, +400g additional pendulum mass
Mp_c = [0.215, 0.425, 0.530, 0.642]  # Kg
Lp_c = [0.314, 0.477, 0.516, 0.546]  # m
I0_c = [7.06e-3, 18.74e-3, 21.89e-3, 24.62e-3]  # Kg m^2

# Other Constants
Mc_c = 1.608  # Kg_c
R_c = 0.16  # Ohms
r_c = 0.0184  # m
Kg_c = 3.71  # ratio (unit-less)
Km_c = 1.68e-2  # V/rad/s
g_c = 9.81  # ms^-2
C_c = 0  # TODO ??

# Value limits
MAX_Ts = 7
MAX_Tr = 3
MAX_Mp = 10
MAX_ess = 0.02
MAX_V = 10
MAX_dVdt = 30


########################################################################################################################
#
#                                                  FUNCTIONS/CLASSES
#
########################################################################################################################


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.array(scipy.linalg.inv(R) @ (B.T @ X))

    eigVals, eigVecs = scipy.linalg.eig(A - B @ K)

    return K, X, eigVals


def identity_mat(n: int):
    return diag(ones(n))


def zero_mat(n: int, m: int):
    return zeros((n, m))


def get_state_space(i: int, offset: float = 0):
    """

    :param i: Parameter set
    :param offset: % offset for parameters
    :return: State space (A, B, C, D)
    """
    # Initialize variables with offset
    multiplier = 1 + offset
    mp = Mp_c[i] * multiplier
    lp = Lp_c[i] * multiplier
    i0 = I0_c[i] * multiplier
    mc = Mc_c * multiplier
    R = R_c * multiplier
    r = r_c * multiplier
    kg = Kg_c * multiplier
    km = Km_c * multiplier
    c = C_c * multiplier
    g = 9.81

    mass_matrix = array([
        [mc + mp, mp * lp],
        [mp * lp, i0 + mp * lp * lp],
    ])
    spring_matrix = array([
        [0, 0],
        [0, -mp * g * lp]
    ])
    damping_matrix = array([
        [c, 0],
        [0, 0]
    ])

    beta = km * kg / (r * R)
    alpha = beta * kg * km / r

    # Initialize A, B for input U=force
    A = zero_mat(4, 4)
    A[:2, 2:] = identity_mat(2)
    A[2:, :2] = -inv(mass_matrix) @ spring_matrix
    A[2:, 2:] = -inv(mass_matrix) @ damping_matrix
    B = zero_mat(4, 1)
    B[2:, 0] = inv(mass_matrix)[:, 0]

    # Change input from U=Force to U=Voltage
    A -= B * alpha @ array([[0, 0, 1, 0]])
    B *= beta
    C = zero_mat(2, 4)
    C[:2, :2] = identity_mat(2)
    D = zero_mat(2, 1)

    return A, B, C, D


def n_gain(A, B, C, K):
    """

    :param A: System dynamics
    :param B: Input matrix
    :param C: Sensor matrix
    :param K: Feedback gains
    :return: N gain
    """
    return -1 / (C @ inv(A - (B @ K)) @ B)[0]


def get_response(system, K, input_t, input_signal=0.1, tracking_gain=None, tracked_var=0):
    """

    :param system: (A, B, C, D)
    :param K: Closed loop gain
    :param input_t: Time array
    :param input_signal: Input signal or step magnitude
    :param tracking_gain:
    :param tracked_var:
    :return: system response (y), controller effort (voltage), dV/dt and N
    """
    A, B, C, D = system
    A_cl = A - B @ K

    if tracking_gain is None:
        N = n_gain(A, B, C[tracked_var, :], K)
    else:
        N = tracking_gain

    if isinstance(input_signal, float) or isinstance(input_signal, int):
        input_u = ones(input_t.shape) * input_signal
    else:
        input_u = input_signal

    B_hat = N * B
    _, y, x = lsim(system=(A_cl, B_hat, C, D), U=input_u, T=input_t, X0=[0]*A.shape[0])
    y = y.T
    voltage = input_signal * N - K @ x.T
    dt = input_t[1] - input_t[0]
    dvdt = diff(voltage) / dt

    return y, voltage, dvdt, N


def plot_response(y, voltage, dvdt, t, fig, title, label):
    """

    :param y: System response
    :param voltage: Controller effort
    :param dvdt: Rate of change of controller effort
    :param t: Time array
    :param fig: Figure number
    :param title: Figure title
    :param label: Label for new figure entry
    """

    if plt.fignum_exists(fig):
        # Create Figure (or use existing figure)
        plt.figure(fig)

        # Displacement
        plt.subplot(2, 2, 1, ylabel='Cart Displacement (m)', xlabel='Time (s)')
        plt.plot(t[:], y[0, :])

        # Angle
        plt.subplot(2, 2, 2, ylabel='Angular Displacement (deg)', xlabel='Time (s)')
        plt.plot(t[:], y[1, :] * 180 / pi, label=label)
        plt.legend(loc="upper right")

        # Voltage
        plt.subplot(2, 2, 3, ylabel='Input Voltage (V)', xlabel='Time (s)')
        plt.plot(t[:], voltage[0, :])

        # dV/dt
        plt.subplot(2, 2, 4, ylabel='dV/dt (V/s)', xlabel='Time (s)')
        plt.plot(t[:-1], dvdt[0, :])
    else:
        # Time values
        t_min = t[0]
        t_max = t[-1]

        # Create Figure (or use existing figure)
        plt.figure(fig)
        plt.title(title)

        # Displacement
        plt.subplot(2, 2, 1, ylabel='Cart Displacement (m)', xlabel='Time (s)')
        plt.ylim(-0.1, 0.2)
        plt.xlim(t_min, t_max)
        plt.axhline(y=0.1 * 1.02, color='k', alpha=0.3, linestyle=':')
        plt.axhline(y=0.1 * 0.98, color='k', alpha=0.3, linestyle=':')
        plt.plot(t[:], y[0, :])

        # Angle
        plt.subplot(2, 2, 2, ylabel='Angular Displacement (deg)', xlabel='Time (s)')
        plt.ylim(-15, 15)
        plt.xlim(t_min, t_max)
        plt.plot(t[:], y[1, :] * 180 / pi, label=label)
        plt.legend(loc="upper right")

        # Voltage
        plt.subplot(2, 2, 3, ylabel='Input Voltage (V)', xlabel='Time (s)')
        plt.ylim(-15, 15)
        plt.plot(t[:], voltage[0, :])
        plt.xlim(t_min, t_max)

        # dV/dt
        plt.subplot(2, 2, 4, ylabel='dV/dt (V/s)', xlabel='Time (s)')
        plt.ylim(-45, 45)
        plt.plot(t[:-1], dvdt[0, :])
        plt.xlim(t_min, t_max)

    # Show figure
    plt.show(block=False)


# Checking functions.  Structured such that a failure message is returned if the check fails, otherwise None


def check_slew(**kwargs):
    if any(abs(val) > 30 for val in kwargs['dvdt'][0, :]):
        return f"FAILURE [max(|dV/dt|) > 30, {max(kwargs['dvdt'][0, :], key=abs):.1f}V/s]"
    return ""


def check_voltage(**kwargs):
    if any(abs(val) > 12 for val in kwargs['v'][0, :]):
        return f"FAILURE [max(|V|) > 12, {max(kwargs['v'][0, :], key=abs):.1f}V]\n"
    return ""


def check_rise_time(**kwargs):
    t_end = t_start = 0
    step_magnitude = max(kwargs['u'])
    for t_i, x_i in zip(kwargs['t'], kwargs['y'][0, :]):
        if x_i <= 0.1 * step_magnitude:
            t_start = t_i
        if x_i >= 0.9 * step_magnitude:
            t_end = t_i
            break

    if t_end - t_start >= 1:
        return f"FAILURE [t_rise > 5s, {t_end - t_start:.1f}s]\n"
    return ""


def check_theta(**kwargs):
    # Maximum |theta| < 10 deg
    if any(abs(val) >= 10 for val in kwargs['y'][1, :]):
        return f"FAILED: max(|theta|)={max(kwargs['y'][1, :], key=abs):.1f}deg\n"
    return ""


def check_settle_time(**kwargs):
    A, B, C, D = kwargs['sys']
    K = kwargs['k']
    N = kwargs['n']
    step_magnitude = max(kwargs['u'])
    x_ss = -inv(A-B@K) @ B * N * step_magnitude
    x_low = x_ss[0, :] * 0.98
    x_high = x_ss[0, :] * 1.02
    t_settled = 0

    for t, x in zip(kwargs['t'], kwargs['y'][0, :]):
        if not x_low <= x <= x_high:
            t_settled = t

    if t_settled >= 5:
        return f"FAILURE [t_rise > 5s, {t_settled:.1f}s]\n"
    return ""


def check_stability(**kwargs):
    A, B, C, D = kwargs['sys']
    K = kwargs['k']
    poles = eig(A - B @ K)
    if any(pole.real >= 0 for pole in poles):
        return f"UNSTABLE: eig={poles}\n"
    return ""


def check_performance(feedback, tracking, t_array, u_array, checks, ss_generate, num_combinations):
    """

    :param feedback: Feedback gains K
    :param tracking: Tracking gain N
    :param t_array: Time array
    :param u_array: Input array r(t)
    :param checks: Assume all checks take **kwargs sys, k, n, y, v, dvdt
    :param ss_generate: Generates state space, takes arguments 0 <= i < num_combinations, uncertainty/100
    :param num_combinations: Number of combinations for ss_generate
    :return: Failure log string
    """
    fail_log = ""

    for j in range(num_combinations):
        for uncertainty in (-10, 5, 0, 5, 10):
            set_log = ""
            sys = ss_generate(j, uncertainty/100)
            y, v, dvdt, _ = get_response(sys, feedback, t_array, u_array, tracking_gain=tracking)

            unstable = check_stability(sys=sys, k=feedback, n=tracking, u=u_array, t=t_array, y=y, v=v, dvdt=dvdt)

            if unstable:
                fail_log += f"Set {i}, Uncertainty {uncertainty}:\n" + unstable

            else:
                for check in checks:
                    set_log += check(sys=sys, k=feedback, n=tracking, u=u_array, t=t_array, y=y, v=v, dvdt=dvdt)

                if set_log:
                    fail_log += f"Set {i}, Uncertainty {uncertainty}:\n" + set_log

    return fail_log


def check_gain_performance(feedback_gains, tracking_gain, t_linspace, plot=0):
    """
    Test chosen feedback gains against design criteria, system combinations, uncertainty and sensor error.
    :param feedback_gains: Array of feedback gains K = [K1, K2, K3, K4]
    :param tracking_gain:
    :param t_linspace:
    :param plot:
    """
    K = feedback_gains
    N = tracking_gain
    step_magnitude = 0.1

    fail_log = ''
    fails = 0

    # Iterate through value combinations, uncertainty levels and +/- uncertainty extremes
    for i in range(4):
        A, B, C, D = get_state_space(i, 0)
        y, V, dVdt, _ = get_response((A, B, C, D), K, t_linspace, step_magnitude, tracking_gain=N)
        plot_response(y, V, dVdt, t_linspace, 99, f'{K}, {N}', f'{i}')

        for uncertainty in (0, 5, 10):
            for sign in (-1, +1):
                # Obtain state space matricies
                A, B, C, D = get_state_space(i, sign*uncertainty/100)
                y, V, dVdt, _ = get_response((A, B, C, D), K, t_linspace, step_magnitude, tracking_gain=N)

                if any(pole.real >= 0 for pole in eig(A-B@K)):
                    fail_log += f'{i+1}, {sign*uncertainty}.  UNSTABLE: eig={eig(A-B@K)}\n'
                    continue

                displacement = y[0, :]
                theta = y[1, :] * 180/pi

                # Rise time < 1s
                t_end = t_start = 0
                for t_i, x_i in zip(t_linspace, displacement):
                    if x_i <= 0.1 * step_magnitude:
                        t_start = t_i
                    if x_i >= 0.9 * step_magnitude:
                        t_end = t_i
                        break

                t_rise = t_end - t_start
                if t_rise >= 1:
                    fail_log += f'{i+1}, {sign*uncertainty}.  FAILED: T_r={t_rise}\n'
                    fails += 1

                # Setting Time < 5s
                # TODO

                # if t_settled >= 5:
                #     fail_log += f'{i}, {sign * uncertainty}.  FAILED: T_s={t_settled}\n'

                # Maximum |theta| < 10 deg
                if any(abs(val) >= 10 for val in theta):
                    fail_log += f'{i+1}, {sign*uncertainty}.  FAILED: max(theta, key=abs)={max(theta, key=abs)}\n'
                    fails += 1

                # Steady-state error < 2%
                x_ss = -inv(A-B@K) @ B * N * step_magnitude
                print(x_ss, x_ss.shape)

                # Maximum |V| < 10
                if any(abs(val) >= 10 for val in V[0, :]):
                    fail_log += f'{i+1}, {sign*uncertainty}.  FAILED: max(V, key=abs)={max(V[0, :], key=abs)}\n'
                    fails += 1

                # Maximum |dV/dt| < 30
                if any(abs(val) >= 30 for val in dVdt[0, :]):
                    fail_log += f'{i+1}, {sign*uncertainty}.  FAILED: max(dVdt, key=abs)={max(dVdt[0, :], key=abs)}\n'
                    fails += 1

                if plot:
                    title = f'K={",".join([f"{k:.2f}" for k in K[0, :]])}, N={N:.2f}'
                    label = f'{i+1}, {sign*uncertainty}'
                    fignum = plot
                    plot_response(y, V, dVdt, t_linspace, fignum, title, label)

    fail_log = f'{fails} Tests failed\n\n' + fail_log if fails else f''

    return fail_log


def obtain_input_nums(prompt_msg: str, type_conversion: type, num_values: int):
    """

    :param prompt_msg: Message for input prompt
    :param type_conversion: Type to convert values to
    :param num_values: number of values expected
    :return: return list of values that meet conditions
    """
    while True:
        try:
            inp = input(prompt_msg).strip()
            if inp == 'q':
                raise KeyboardInterrupt
            for sep in (', ', ' ', ','):
                tmp = inp.split(sep)
                if len(tmp) == num_values:
                    return [type_conversion(x) for x in tmp]
        except ValueError:
            continue


# Run if pendulum.py is executed, but not when imported
if __name__ == '__main__':

    try:
        i, = obtain_input_nums('Choose system (1-4)\n> ', int, 1)
        # Choose system to model for
        A, B, C, D = get_state_space(i - 1)
        t_min, t_max, n_points = obtain_input_nums('Define T linspace (min,max,n)\n> ', int, 3)
        t_linspace = linspace(t_min, t_max, n_points)

        while True:

            # # Pole Placement
            # poles = obtain_input_nums('Enter 4 poles\n> ', complex, 4)
            # full_state = place_poles(A, B, poles)
            # K_pp = full_state.gain_matrix
            # y, voltage, dvdt, N = get_response((A, B, C, D), array(K_pp), t_linspace)
            # label = f'Kpp={",".join([f"{k:.2f}" for k in K_pp[0]])} N={N:.2f}'
            # plot_response(y, voltage, dvdt, t_linspace, i, f'', label)
            # results = check_gain_performance(K_pp, N, t_linspace, plot=0)
            # print(results)

            # LQR
            Q1, Q2, Q3, Q4, R1 = obtain_input_nums('Enter Q weights followed by R value (4 then 1)\n> ', float, 5)
            Q = diag([Q1, Q2, Q3, Q4])
            R = array([[R1]])
            K_lqr, X, E = lqr(A, B, Q, R)
            y, v, dVdt, N = get_response((A, B, C, D), K_lqr, t_linspace)
            label = f'Klqr={",".join([f"{k:.2f}" for k in K_lqr[0, :]])} N={N:.2f}'
            plot_response(y, v, dVdt, t_linspace, i, f'', label)
            plt.close(99)
            results = check_gain_performance(K_lqr, N, t_linspace, plot=0)
            print(results)

    except (KeyboardInterrupt, EOFError):
        pass
