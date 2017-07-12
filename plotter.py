save_plot_to_file = True

from environment.pose import Pose
import numpy as np
from datetime import datetime
import matplotlib

if save_plot_to_file:
    matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

__author__ = 'philippe'


class Plotter:
    # TODO: Clean
    def __init__(self, exp_id, x_limits, y_limits, n_steps, cap_checkpoints=False,
                 dynamic_fn=False, plot_final_belief=False, plot_color_traj=True, plot_acc_rew=False,
                 plot_belief_evo=False):
        self.exp_id = exp_id
        self.x_min = x_limits[0]
        self.x_max = x_limits[1]
        self.y_min = y_limits[0]
        self.y_max = y_limits[1]
        self.agt_chckpt_x = []
        self.agt_chckpt_y = []
        self.agt_step_x = []
        self.agt_step_y = []
        self.agt_step_t = []
        self.agt_start = []
        self.agt_step_obs = []
        self.agt_bel = None
        self.grid_res = 0
        self.grid_evo = None
        self.bel_evo_m = []
        self.bel_evo_std = []
        self.n_steps = n_steps
        self.dynamic_fn = dynamic_fn
        self.b_plot_final_belief = plot_final_belief
        self.b_plot_color_traj = plot_color_traj
        self.b_plot_acc_rew = plot_acc_rew
        self.b_plot_belief_evo = plot_belief_evo

        self.cap_checkpoints = cap_checkpoints
        self.save_to_file = save_plot_to_file
        self.acc_rew = None

        # Compute number of plots
        self.nb_plots = 0
        if self.b_plot_final_belief and not self.dynamic_fn:
            self.nb_plots += 2
        if self.b_plot_color_traj:
            self.nb_plots += 1
        if self.b_plot_acc_rew:
            self.nb_plots += 1
        self.subplot_conf = [2 if self.nb_plots > 2 else 1, 2 if self.nb_plots > 1 else 1]
        self.subplot_offset = 1

    def add_agent_start(self, pose, obs):
        self.agt_start = Pose.to_xy_array(pose)
        self.add_agent_step(pose, obs)

    def add_agent_pose(self, pose):
        if self.cap_checkpoints:
            self.agt_chckpt_x.append(pose.x)
            self.agt_chckpt_y.append(pose.y)

    def add_agent_step(self, pos, obs):
        if self.b_plot_color_traj:
            self.agt_step_x.append(pos.x)
            self.agt_step_y.append(pos.y)
            self.agt_step_t.append(pos.t)
            self.agt_step_obs.append(obs)

        # Capture belief evolution
        if self.b_plot_belief_evo and self.agt_bel is not None:
            if self.dynamic_fn:
                self.agt_bel = None
                print 'Warning: Not capturing belief evolution for dynamic objective function'
            else:
                z_mean, z_std = self.agt_bel.estimate(self.grid_evo)
                self.bel_evo_m.append(np.reshape(z_mean, (self.grid_res, self.grid_res)))
                self.bel_evo_std.append(np.reshape(z_std, (self.grid_res, self.grid_res)))

    def set_acc_reward(self, acc_rew):
        self.acc_rew = acc_rew

    def set_agent_beleif(self, agt_bel):
        self.agt_bel = agt_bel

    def display(self):
        self.subplot_offset = 1
        fig = plt.figure(figsize=(14, 10))
        plt.subplots_adjust(left=0.05, bottom=0.08, top=0.92, right=0.95, wspace=0.15, hspace=0.15)
        if self.b_plot_color_traj:
            self.plot_color_trajectory(fig)
            self.subplot_offset += 1
        if self.b_plot_acc_rew:
            self.plot_accumulated_reward(fig)
            self.subplot_offset += 1
        if self.b_plot_final_belief and not self.dynamic_fn:
            self.plot_final_belief(self.agt_bel, res=32)

        # Save to file or show figure
        if self.subplot_offset != 1:
            if self.save_to_file:
                plt.savefig(
                    'res/behaviour_{}'.format(self.exp_id) + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.png')
                # , dpi=2*fig.dpi)
                plt.clf()
            else:
                plt.show()

        if self.b_plot_final_belief and self.dynamic_fn:
            self.plot_final_belief(self.agt_bel, res=16, time_limits=[0, 10], time_res=40)
        if self.b_plot_belief_evo:
            self.plot_belief_evolution()

    def plot_color_trajectory(self, fig):
        if len(self.agt_step_x) == 0:
            print 'Warning: Can\'t print trajectory without data.'
            return
        jet = plt.get_cmap('jet')
        c_norm = mcolors.Normalize(vmin=0.1, vmax=1.0)
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
        color_factor = 1.0

        # If obejctive function is dynamic
        if self.dynamic_fn:
            color_factor = 10.0

        # Setup trajctory color scheme, figure and axes
        color_val = [scalar_map.to_rgba(o * color_factor) for o in self.agt_step_obs]
        ax = plt.subplot(self.subplot_conf[0], self.subplot_conf[1], self.subplot_offset)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        # Plot starting point
        plt.title(self.exp_id)
        plt.plot(self.agt_start[0], self.agt_start[1], 'ro')

        # Plot checkpoints
        if self.cap_checkpoints:
            plt.plot(self.agt_chckpt_x, self.agt_chckpt_y, 'b*')

        # Plot trajectory
        for i in range(len(self.agt_step_x) - 1):
            plt.plot([self.agt_step_x[i], self.agt_step_x[i + 1]], [self.agt_step_y[i], self.agt_step_y[i + 1]],
                     color=color_val[i], linewidth=1.5)

    def gather_belief_data(self, res=8):
        grid_x, grid_y = np.meshgrid(np.linspace(self.x_min, self.x_max, res),
                                     np.linspace(self.y_min, self.y_max, res))
        self.grid_evo = np.concatenate((np.reshape(grid_x, (res ** 2, 1)), np.reshape(grid_y, (res ** 2, 1))), axis=1)
        self.grid_res = res

    def plot_belief_evolution(self):
        numframes = len(self.bel_evo_m)
        if numframes == 0:
            print 'Warning: Can\'t print belief evolution without data.'
            return

        fig = plt.figure()
        plt.title(self.exp_id)
        dt = float(self.n_steps) / float(numframes)
        grid_x = np.reshape(self.grid_evo[:, 0], (self.grid_res, self.grid_res))
        grid_y = np.reshape(self.grid_evo[:, 1], (self.grid_res, self.grid_res))
        self.__anim_plot(fig, grid_x, grid_y, self.bel_evo_m, numframes, dt)

    def plot_final_belief(self, belief, res=8, time_limits=None, time_res=20):
        grid_x, grid_y = np.meshgrid(np.linspace(self.x_min, self.x_max, res), np.linspace(self.y_min, self.y_max, res))

        # If obejctive function is static
        if time_limits is None:
            x_test = np.concatenate((np.reshape(grid_x, (res ** 2, 1)), np.reshape(grid_y, (res ** 2, 1))), axis=1)
            z_mean, z_std = belief.estimate(x_test)
            z_mean = np.reshape(z_mean, (res, res))
            z_std = np.reshape(z_std, (res, res))

            plt.title(self.exp_id)
            plt.subplot(self.subplot_conf[0], self.subplot_conf[1], self.subplot_offset)
            plt.contourf(grid_x, grid_y, z_mean, 64, alpha=.75, cmap='jet')
            # plt.contour(grid_x, grid_y, z_mean, 32, colors='black', linewidth=.2)
            plt.xlim(self.x_min, self.x_max)
            plt.ylim(self.y_min, self.y_max)

            plt.subplot(self.subplot_conf[0], self.subplot_conf[1], self.subplot_offset + 1)
            plt.contourf(grid_x, grid_y, z_std, 64, alpha=.75, cmap='jet')
            # plt.contour(grid_x, grid_y, z_std, 32, colors='black', linewidth=.2)
            plt.xlim(self.x_min, self.x_max)
            plt.ylim(self.y_min, self.y_max)

        else:
            # Dynamic objective function
            grid_t = np.linspace(time_limits[0], time_limits[1], time_res)
            bel_m = [0] * time_res
            bel_std = [0] * time_res
            for k in range(time_res):
                x_space = np.concatenate((np.reshape(grid_x, (res ** 2, 1)), np.reshape(grid_y, (res ** 2, 1))), axis=1)
                x_test = np.concatenate((x_space, grid_t[k] * np.ones((res ** 2, 1))), axis=1)
                z_mean, z_std = belief.estimate(x_test)
                bel_m[k] = np.reshape(z_mean, (res, res))
                bel_std[k] = np.reshape(z_std, (res, res))
            fig = plt.figure()
            self.__anim_plot(fig, grid_x, grid_y, bel_m, time_res, grid_t[1] - grid_t[0])

    def __anim_plot(self, fig, grid_x, grid_y, all_data, numframes, dt):
        def update_contour_plot(i, data, ax, xi, yi):
            ax.cla()
            im = ax.contourf(xi, yi, data[i], 15, alpha=.75, cmap='jet')
            plt.title('time: {:.2f}s'.format(i * dt))
            return im,

        ax = fig.gca()

        if self.save_to_file:
            FFMpegWriter = animation.writers['ffmpeg']
            metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
            writer = FFMpegWriter(fps=numframes / dt, metadata=metadata)
            with writer.saving(fig, "res/writer_test.mp4", numframes):
                for i in range(numframes):
                    update_contour_plot(i, all_data, ax, grid_x, grid_y)
                    writer.grab_frame()
        else:
            ani = animation.FuncAnimation(fig, update_contour_plot, frames=xrange(numframes),
                                          fargs=(all_data, ax, grid_x, grid_y), interval=1000 * dt)
            plt.show()

    def plot_accumulated_reward(self, fig):
        plt.subplot(self.subplot_conf[0], self.subplot_conf[1], self.subplot_offset)
        plt.plot(range(len(self.acc_rew)), self.acc_rew)
        plt.xlabel('Time steps')
        plt.ylabel('Accumulated reward')
