import os
from numpy import array, zeros, diff
import matplotlib.pyplot as plt
from time import sleep


def get_data(file_path, var_types, delimiter=','):
    with open(file_path, 'r') as file:
        d_array = None
        for i, v_type in enumerate(var_types):
            line = file.readline().strip(', \n')
            line_data = array([v_type(val) for val in line.split(delimiter)])
            if d_array is None:
                d_array = zeros((line_data.shape[0], len(var_types)))
            d_array[:, i] = line_data[:]
    return d_array


def get_paths(directory, extension='txt'):
    file_paths = []
    for file_name in os.listdir(directory):
        if file_name[-(len(extension)+1):] == f'.{extension}':
            file_paths.append(file_name)
    return file_paths


def main():
    cwd = os.getcwd()
    var_dict = {
        f'{cwd}/Lab-Cart/': [
            'Time',
            'Cart 1 Position (m)', 'Cart 1 Kp', 'Cart 1 Velocity (m/s)', 'Cart 1 Kv',
            'Cart 2 Position (m)', 'Cart 2 Kp', 'Cart 2 Velocity (m/s)', 'Cart 2 Kv',
            'Cart 3 Position (m)', 'Cart 3 Kp', 'Cart 3 Velocity (m/s)', 'Cart 3 Kv',
            'Cart Motor Voltage (V)', 'Input r(t)', 'Raw Motor Voltage (V)'
        ],
        f'{cwd}/Lab-Pendulum/': [
            'Time',
            'Cart Position (m)', 'Cart Kp', 'Cart Velocity (m/s)', 'Cart Kv',
            'N gain', 'Input r(t)', 'Pendulum Kp', 'Pendulum Angle (rad)',
            'Pendulum Angular Velocity (rad/s)', 'Pendulum Kv', 'Raw Motor Voltage (V)'
        ],
    }
    fig_groups = {
        f'{cwd}/Lab-Cart/': {
            211: ('Cart Position', 'Displacement (m)', [(0, 'Cart 1 Position (m)'), (4, 'Cart 2 Position (m)'), (8, 'Cart 3 Position (m)'),
                  (13, 'Input r(t)')]),
            212: ('Cart and Raw Voltage', 'Voltage (V)', [(12, 'Cart Motor Voltage (V)'), (14, 'Raw Motor Voltage (V)')])
        },
        f'{cwd}/Lab-Pendulum/': {
            311: ('Cart Position', 'Displacement (m)', [(0, 'Cart Position (m)'), (5, 'Input r(t)')]),
            312: ('Pendulum Position', 'Angle (rad)', [(7, 'Pendulum Angle (rad)')]),
            313: ('Motor Voltage', 'Voltage (V)', [(10, 'Raw Motor Voltage (V)')])
        }
    }
    fig = 1
    for directory in var_dict:
        paths = get_paths(directory, extension='txt')
        for fig, file_name in enumerate(paths, start=fig):
            file_path = directory + file_name
            data_array = get_data(file_path, [float] * len(var_dict[directory]))
            time = data_array[:, 0]
            data = data_array[:, 1:]
            print(file_path)

            # Plot Data
            plt.figure(fig)
            for subplot in sorted(fig_groups[directory]):
                title, ylabel, variables = fig_groups[directory][subplot]
                plt.subplot(subplot)
                plt.title(title)
                plt.ylabel(ylabel)
                plt.xlabel('Time (s)')

                for j, var in variables:
                    plt.plot(time, data[:, j], label=var[:var.rfind('(') if var != 'Input r(t)' else len(var)])
                    plt.grid(True)

                plt.legend(loc="upper right")
                # if var == 'Raw Motor Voltage (V)':
                #     color = 'tab:purple'
                #     plt.twinx()
                #     dt = time[1] - time[0]
                #     plt.ylabel('Slew Rate (V/s)', color=color)
                #     plt.plot(time[:-1], diff(data[:, j])/dt, label='dV/dt', color=color)
                #     plt.legend(loc="lower right")

            plt.tight_layout()
            plt.show(block=True)
        fig += 1


if __name__ == '__main__':
    main()
