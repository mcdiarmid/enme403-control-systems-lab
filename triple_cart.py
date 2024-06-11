"""
ENME403 Triple-Cart Lab 2019S1
Author: Campbell McDiarmid

"""

########################################################################################################################
#
#                                                    IMPORTS
#
########################################################################################################################


from numpy import array, zeros, ones, pi, exp, diag, linspace
from numpy.linalg import inv
from numpy.linalg import eigvals as eig
from scipy.signal import tf2ss, ss2tf, step, impulse, place_poles, lsim
import matplotlib.pyplot as plt

from pendulum import obtain_input_nums, identity_mat, zero_mat, n_gain, get_response, lqr


########################################################################################################################
#
#                                                    CONSTANTS
#
########################################################################################################################


m1_c = 1.608
mass_combs = [0.75, 1.25]
spring_const_combs = [175, 400, 800]
damping = [0, 3.68, 3.68]
alpha = 12.45
Km = 0.00176
Kg = 3.71
R_c = 1.4
r = 0.0184
V_max = 12

system_combinations = [(m1_c, m2, m3, k) for m2 in mass_combs for m3 in mass_combs for k in spring_const_combs]


########################################################################################################################
#
#                                                  FUNCTIONS/CLASSES
#
########################################################################################################################


def generate_input(period, duration, points, amplitude):
    return array([amplitude if (2 * duration * i / points) // period % 2 else 0 for i in range(points)])


def plot_combination_response():
    # Initialize A, B1, C1, D1
    A = zero_mat(6, 6)
    A[:3, 3:] = identity_mat(3)
    B = zero_mat(6, 1)
    C = zero_mat(3, 6)
    C[:3, :3] = identity_mat(3)
    D = zero_mat(1, 6)

    # Create other constant matricies
    C_damping = diag(damping)

    # Iterate through all combinations
    for m1, m2, m3, k in system_combinations:
        # Populate matricies with the combination's values
        M_mass = diag([m1, m2, m3])
        K_spring = array([
            [+k, -k, 0],
            [-k, +k + k, -k],
            [0, -k, +k]
        ])
        A[3:, :3] = -inv(M_mass) @ K_spring
        A[3:, 3:] = -inv(M_mass) @ C_damping
        A[3, 3] -= Km * Km * Kg * Kg / (m1 * R_c * r * r)
        B[3, 0] = alpha * Km * Kg / (m1 * R_c * r)


def main():
    # Initialize A, B1, C1, D1
    A = zero_mat(6, 6)
    A[:3, 3:] = identity_mat(3)
    B = zero_mat(6, 1)
    # C = zero_mat(3, 6)
    # C[:3, :3] = identity_mat(3)
    C = zero_mat(3, 6)
    C[:3, :3] = identity_mat(3)
    D = zero_mat(3, 1)

    # Create other constant matricies
    C_damping = diag(damping)

    m1, m2, m3, k = system_combinations[6]

    M_mass = diag([m1, m2, m3])
    K_spring = array([
        [+k, -k, 0],
        [-k, +k + k, -k],
        [0, -k, +k]
    ])
    A[3:, :3] = -inv(M_mass) @ K_spring
    A[3:, 3:] = -inv(M_mass) @ C_damping
    A[3, 3] -= Km * Km * Kg * Kg / (m1 * R_c * r * r)
    B[3, 0] = alpha * Km * Kg / (m1 * R_c * r)

    try:
        # Choose system to model for
        t_min, t_max, n_points = obtain_input_nums('Define T linspace (min,max,n)\n> ', int, 3)
        t_linspace = linspace(t_min, t_max, n_points)

        period, = obtain_input_nums('Define input period\n> ', float, 1)
        U = generate_input(period, t_max - t_min, n_points, 0.5)

        while True:
            # Generate input signal array

            # # Pole Placement
            # poles = obtain_input_nums('Enter 6 poles\n> ', complex, 6)
            # full_state = place_poles(A, B, poles)
            # K_pp = full_state.gain_matrix
            # y, voltage, dvdt, N = get_response((A, B, C, D), array(K_pp), t_linspace, input_signal=U, tracked_var=2)
            # label = f'Kpp={",".join([f"{k:.2f}" for k in K_pp[0]])} N={N:.2f}'
            # print(label)
            # plt.plot(t_linspace, y.T)
            # plt.show()
            # plt.plot(t_linspace, voltage.T, t_linspace[:-1], dvdt.T)
            # plt.show()

            # LQR
            lqr_vals = obtain_input_nums('Enter Q weights followed by R value (6 then 1)\n> ', float, 7)
            Q = diag(lqr_vals[:6])
            R_cost = array([[lqr_vals[-1]]])
            K_lqr, x, e = lqr(A, B, Q, R_cost)
            y, voltage, dvdt, N = get_response((A, B, C, D), array(K_lqr), t_linspace, input_signal=U, tracked_var=2)
            title = f'Klqr={",".join([f"{k:.2f}" for k in K_lqr[0, :]])} N={N:.2f}'

            plt.close(1)
            plt.close(2)

            plt.figure(1)
            plt.title(title)
            plt.plot(t_linspace, y.T, t_linspace, U, 'k:')
            plt.legend(['Cart1', 'Cart2', 'Cart3'])
            plt.ylabel('Displacement (m)')
            plt.xlabel('Time (s)')
            plt.ylim((-0.5, 1.0))
            plt.show(block=False)

            plt.figure(2)
            plt.title('Controller Response')
            plt.subplot(211)
            plt.plot(t_linspace, voltage[0, :])
            plt.ylabel('Voltage (V)')
            plt.subplot(212)
            plt.plot(t_linspace[:-1], dvdt[0, :])
            plt.ylabel('Slew (V/s)')
            plt.xlabel('Time (s)')
            plt.show(block=False)

    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == '__main__':
    main()
