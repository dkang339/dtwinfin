'''
Thermal fin digital twin simulator

This class simulates digital twin under varying heat fluxes and saves the resulting animation.
The steps are:
1. True heat flux changes over time
2. Error-aware RB model inversely identifies the heat flux
3. FE model monitors the temperature field
4. Control actions are applied when thresholds are exceeded and FE model monitors the corresponding field
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.gridspec import GridSpec
import os

from inverse_id import (
    read_sensor, identify_with_error, identify,
    monitor, control_cooling, plot_field
)


class DTwinSimulator:
    '''
    digital twin simulator for thermal fin
    '''

    def __init__(self, truth_problem, reduced_problem, kriging_model,
                 sensor_loc, output_dir="./dtwin_results"):
        '''
        initialize simulator.

        inputs:
            truth_problem (Subfin class instance): FE problem
            reduced_problem (Subfin class instance): RB problem
            kriging_model (Kriging class instance): trained Kriging model
            sensor_loc (list): sensor vertex indices
            output_dir (str): output file directory
        '''
        self.truth_problem = truth_problem
        self.reduced_problem = reduced_problem
        self.kriging = kriging_model
        self.sensor_loc = sensor_loc
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True) # create output directory
        self.base_mu = [2.0, 1.0, 5.0]  # [Bi_p, Bi_s, q]

        # control thresholds
        self.low_threshold = 0.5
        self.high_threshold = 1.5

        # get mesh for plotting
        mesh = truth_problem.V.mesh()
        self.mesh = mesh
        coords = mesh.coordinates()
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        cells = mesh.cells()
        self.triang = tri.Triangulation(self.x, self.y, cells)
        self.vmin, self.vmax = 0.0, 1.9 # colorbar limits

    def generate_heat_flux(self, t_steps):
        '''
        generate time-varying heat flux.

        inputs:
            t_steps (int): number of time steps

        outputs:
            q (array): heat flux values over time
        '''
        t = np.linspace(0, 1, t_steps)

        q = np.zeros(t_steps)
        q[:t_steps//8] = 2.0 # low 
        q[t_steps//8:3*t_steps//8] = 10.0 # high
        q[3*t_steps//8:5*t_steps//8] = 5.0 # normal
        q[5*t_steps//8:7*t_steps//8] = 9.0 # high
        q[7*t_steps//8:] = 3.0 # low

        return q

    def run_simulation(self, t_steps=10, compare_fe=False):
        '''
        run simulation and save data.

        inputs:
            t_steps (int): number of time steps
            compare_fe (bool): if True, run FE-based identification for comparison

        Returns:
            history (dict): simulation data (time, system parameters, max temp)
        '''
        import time as time

        q_true = self.generate_heat_flux(t_steps)

        # initialize history data
        history = {
            'step': np.arange(t_steps),
            'q_true': q_true,
            'q_estimated': np.zeros(t_steps),
            'max_temp': np.zeros(t_steps),
            'max_temp_after_control': np.zeros(t_steps),
            'Bi_p': np.zeros(t_steps),
            'Bi_s': np.zeros(t_steps),
            'control_action': [],
            # initialize arrays for time spent at each step (seconds)
            'time_sensor': np.zeros(t_steps),
            'time_identify': np.zeros(t_steps),
            'time_monitor': np.zeros(t_steps),
            'time_control': np.zeros(t_steps),
            # initialize arrays for cumulative time of each step (seconds)
            'real_time_true': np.zeros(t_steps),
            'real_time_identified': np.zeros(t_steps),
            'real_time_monitored': np.zeros(t_steps),
            'real_time_controlled': np.zeros(t_steps),
            # initialize arrays for time spent for FE-based identification
            'time_identify_fe': np.zeros(t_steps) if compare_fe else None,
            'frames': []
        }

        current_mu = list(self.base_mu)
        cumulative_time = 0.0

        print(f"Running simulation: {t_steps} steps")
        print("-" * 60)

        for step in range(t_steps):
            # track time when q changes
            history['real_time_true'][step] = cumulative_time
            print(f"step {step+1}/{t_steps} (t={cumulative_time:.4f}s: true q = {q_true[step]:.2f})")

            # ----- 1. get sensor measurement -----
            t_start = time.time()
            y_true, u_true = read_sensor(
                truth_problem=self.truth_problem,
                mu=current_mu,
                q_value=q_true[step],
                sensor_loc=self.sensor_loc
            )

            # ----- 2. identify heat flux using error-aware RB based twin -----
            t_start = time.time()
            id_mu, id_time = identify_with_error(
                problem=self.reduced_problem,
                sensor_loc=self.sensor_loc,
                y_true=y_true,
                online_mu=current_mu,
                kriging=self.kriging,
                initial_guess=5.0,
                return_time=True
            )
            history['q_estimated'][step] = id_mu[2]
            history['time_identify'][step] = id_time
            cumulative_time += id_time
            history['real_time_identified'][step] = cumulative_time

            # compare with FE-based identification
            if compare_fe:
                _, fe_time = identify(
                    problem=self.truth_problem,
                    sensor_loc=self.sensor_loc,
                    y_true=y_true,
                    online_mu=current_mu,
                    initial_guess=5.0,
                    rb=False,
                    return_time=True
                )
                history['time_identify_fe'][step] = fe_time
                print(f"error-aware RB-based twin speedup: {fe_time/id_time}x faster")

            # ----- 3. monitor with identified parameters -----
            monitor_mu = list(current_mu)
            monitor_mu[2] = id_mu[2]
            t_start = time.time()
            u_monitor, max_temp = monitor(self.truth_problem, monitor_mu, plot=False) # monitor using FE model
            monitor_time = time.time() - t_start
            history['time_monitor'][step] = monitor_time
            cumulative_time += monitor_time
            history['real_time_monitored'][step] = cumulative_time
            history['max_temp'][step] = max_temp

            # ----- 4. apply control action -----
            t_start = time.time()
            mu_control, u_control, max_temp_control, _, action_state = control_cooling(
                truth_problem=self.truth_problem,
                current_mu=monitor_mu,
                max_temp=max_temp,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                plot=False
            )
            control_time = time.time() - t_start
            history['time_control'][step] = control_time
            cumulative_time += control_time
            history['real_time_controlled'][step] = cumulative_time

            history['control_action'].append(action_state)
            history['Bi_p'][step] = mu_control[0]
            history['Bi_s'][step] = mu_control[1]
            history['max_temp_after_control'][step] = max_temp_control

            # update parameters for next iteration
            current_mu = mu_control

            # save frame data for visualization
            history['frames'].append({
                'u_true': u_true.compute_vertex_values(self.mesh).copy(), # extract solution to plot
                'u_monitor': u_monitor.compute_vertex_values(self.mesh).copy(),
                'u_control': u_control.compute_vertex_values(self.mesh).copy()
            })

        # print simulation time summary
        self._print_time_summary(history, compare_fe)

        return history

    def _print_time_summary(self, history, compare_fe):
        '''
        print simulation time summary.
        '''
        id_times = history['time_identify']
        total_time = history['real_time_controlled'][-1]

        print("=" * 60)
        print("Time summary")
        print(f"total simulation time: {total_time:.3f} s")
        print(f"\nidentification:")
        print(f"  mean:   {np.mean(id_times)*1000:.2f} ms")
        print(f"  std:    {np.std(id_times)*1000:.2f} ms")

        if compare_fe and history['time_identify_fe'] is not None:
            fe_times = history['time_identify_fe']
            print(f"\nFE-based identification:")
            print(f" mean: {np.mean(fe_times)*1000:.2f} ms")
            print(f" error-aware RB-based twin speedup: {np.mean(fe_times)/np.mean(id_times):.1f}x faster")
        print("=" * 60)

    def create_gif(self, history, step, fig=None):
        '''
        create digital twin animation.
        '''
        if fig is None:
            fig = plt.figure(figsize=(14, 10)) # create new fig
        else:
            fig.clear() # reset fig

        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 0.8, 0.8],
                      hspace=0.35, wspace=0.3)

        frame_data = history['frames'][step]
        levels = np.linspace(self.vmin, self.vmax, 50)
        norm = plt.Normalize(self.vmin, self.vmax)

        # ----- top: temperature fields -----
        # true field
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.tricontourf(self.triang, frame_data['u_true'],
                        levels=levels, cmap='inferno', norm=norm)
        ax1.set_title(f"True (q={history['q_true'][step]:.3f})", fontsize=11)
        ax1.set_aspect('equal')
        ax1.axis('off')

        # field monitored by error-aware RB-based twin
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.tricontourf(self.triang, frame_data['u_monitor'],
                        levels=levels, cmap='inferno', norm=norm)
        ax2.set_title(f"error-aware RB digital twin (q={history['q_estimated'][step]:.3f})", fontsize=11)
        ax2.set_aspect('equal')
        ax2.axis('off')

        # field after control
        ax3 = fig.add_subplot(gs[0, 2])
        tcf3 = ax3.tricontourf(self.triang, frame_data['u_control'],
                               levels=levels, cmap='inferno', norm=norm)
        action = history['control_action'][step]
        color = 'red' if action == "Cooling increased" else ('blue' if action == "Cooling reduced" else 'green')
        ax3.set_title(f"After control [{action}]", fontsize=11, color=color)
        ax3.set_aspect('equal')
        ax3.axis('off')

        # add colorbar
        cbar_ax = fig.add_axes([0.92, 0.65, 0.015, 0.25])
        cbar = fig.colorbar(tcf3, cax=cbar_ax)
        cbar.set_label('Temperature (dimensionless)', fontsize=10)

        # ----- middle: heat flux identification -----
        ax4 = fig.add_subplot(gs[1, :2])
        n_steps = len(history['step'])
        t_true_end = min(step + 2, n_steps)
        t_true = history['real_time_true'][:t_true_end]
        t_identified = history['real_time_identified'][:step+1]
        t_controlled = history['real_time_controlled'][:step+1]
        total_time = history['real_time_controlled'][-1]
        ax4.step(t_true, history['q_true'][:t_true_end], 'b-', linewidth=2,
                 where='post', label='True heat flux')
        # plot identified heat flux
        # heat flux is plotted reflecting the identification time
        ax4.plot(t_identified, history['q_estimated'][:step+1],
                'r-', linewidth=2, marker='o', markersize=4,
                label='Identified heat flux')
        ax4.scatter(t_identified[-1], history['q_estimated'][step], # highlight current step
                    color='red', zorder=5)
        ax4.set_xlim(-0.05 * total_time, total_time * 1.05)
        ax4.set_ylim(0, 11)
        ax4.set_xlabel('Time (seconds)', fontsize=10)
        ax4.set_ylabel('Heat flux', fontsize=10)
        ax4.set_title('Heat flux Identification', fontsize=11)
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True)
        # add annotation
        t_true_temp = history['real_time_true'][step]
        t_id_temp = history['real_time_identified'][step]
        id_temp = t_id_temp - t_true_temp
        error = abs(history['q_true'][step] - history['q_estimated'][step])
        ax4.text(0.02, 0.95,
                f"t={t_id_temp:.3f}s\nID Error: {error:.4f}\nID time: {id_temp*1000:.1f}ms",
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # simulation time breakdown
        ax_lat = fig.add_subplot(gs[1, 2])
        steps_so_far = history['step'][:step+1]
        # plot identify + monitor + control times
        id_ms = history['time_identify'][:step+1] * 1000
        mon_ms = history['time_monitor'][:step+1] * 1000
        ctrl_ms = history['time_control'][:step+1] * 1000
        ax_lat.bar(steps_so_far, id_ms, color='blue', alpha=0.7, label='Identification')
        ax_lat.bar(steps_so_far, mon_ms, bottom=id_ms, color='orange', alpha=0.7, label='Monitoring')
        ax_lat.bar(steps_so_far, ctrl_ms, bottom=id_ms+mon_ms, color='green', alpha=0.7, label='Control + monitoring')
        ax_lat.set_xlim(-0.5, len(history['step']) - 0.5)
        ax_lat.set_xlabel('Timestep index', fontsize=10)
        ax_lat.set_ylabel('Time (ms)', fontsize=10)
        ax_lat.set_title('Time breakdown', fontsize=11)
        ax_lat.legend(loc='upper right', fontsize=7)
        ax_lat.grid(True, alpha=0.3, axis='y')

        # ----- bottom: temperature monitoring and control -----
        ax5 = fig.add_subplot(gs[2, :3])
        ax5.axhline(y=self.high_threshold, color='r', linestyle='--',
                    label='High threshold')
        ax5.axhline(y=self.low_threshold, color='b', linestyle='--',
                    label='Low threshold')
        ax5.fill_between([0, total_time], self.low_threshold, self.high_threshold,
                         alpha=0.1, color='green', label='Normal range')
        # plot max temperature at before and after control
        t_monitored = history['real_time_monitored'][:step+1]
        ax5.plot(t_monitored, history['max_temp'][:step+1], 'k-', linewidth=2,
                 marker='s', markersize=4, label='Before control')
        ax5.plot(t_controlled, history['max_temp_after_control'][:step+1], 'g-', linewidth=2,
                 marker='^', markersize=4, label='After control')
        ax5.scatter(t_monitored[-1], history['max_temp'][step], color='black', zorder=5)
        ax5.set_xlim(-0.05 * total_time, total_time * 1.05)
        ax5.set_ylim(0, 2.7)
        ax5.set_xlabel('Time (seconds)', fontsize=10)
        ax5.set_ylabel('Max temperature', fontsize=10)
        ax5.set_title('Monitoring temperature', fontsize=11)
        ax5.legend(loc='upper right', fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.3)

        # set plot title
        current_time = history['real_time_controlled'][step]
        fig.suptitle(f'Thermal fin digital twin | t = {current_time:.3f}s',
                     fontsize=14, fontweight='bold', y=0.98)

        return fig

    def save_gif(self, history, filename="simulation.gif",
                 duration=700, loop=0):
        '''
        generate and save gif.
        '''
        from PIL import Image
        import io

        frames = []
        fig = plt.figure(figsize=(14, 10))
        n_step = len(history['step'])

        for step in range(n_step):
            self.create_gif(history, step, fig)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

            print(f"Frame {step+1}/{n_step}")
        plt.close(fig)

        output_path = os.path.join(self.output_dir, filename)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop
        )
        return output_path