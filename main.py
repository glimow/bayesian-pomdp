import plotter
import math
from datetime import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
from environment.ActionProvider import ActionProvider
import agent
from environment.pose import Pose
from environment.world import World

__author__ = 'philippe'


# TODO: Why do all planners get stuck to the top? (Doesn't happen if obj fun is noisy (var=0.2))
# TODO: MKCF_FTS_cont* is probably bugged
# TODO: Fix dynamic objective funtion
# TODO: Get belief's 'optimize' working
def run_experiment(exp_id, *config):
    """
    Runs experiments. Experiment configuration defined at the bottom of the file.
    :param exp_id: Experience name (used for plotting purposes)
    :param config: a configuration tupple.
    :return: a list of accumulated rewards.
    """
    obj_fun_type = config[5]
    n_steps = config[1]
    acc_rew = [0]

    # Create the action provider
    if config[2] == 'FTS' or config[2] == 'MCTS_disc' or config[2] == 'random':
        nparams = 0
        action_type = 'disc_spline'
    elif config[2] == 'MCTS_cont' and not config[3][3]:
        nparams = 4  # 4 or 5 for continuous splines
        action_type = 'cont_spline'
    elif config[2] == 'MKCF_FTS_cont' or config[2] == 'MKCF_FTS_cont*' or (config[2] == 'MCTS_cont' and config[3][3]):
        nparams = 3  # 3 or 4 for kernel trajectories
        action_type = 'kernel_traj'
    else:
        raise Exception('Unknown configuration: couldn\'t infer action type.')

    boundaries = [[0, 6], [0, 6]]
    act_prov = ActionProvider(boundaries, nparams, action_type, obj_fun_type == 'dynamic')

    # Creat the World
    pose = Pose(1, 1, 0)
    world = World(pose, obj_fun_type, act_prov, obs_noise_var=0.2)

    # Get the first observation
    samp_0 = world.objective_function(world, pose)

    # Create agent and give it a first observation
    agt = agent.Agent(act_prov, world.pose, samp_0, obj_fun_type, config[4], config[2], config[3])

    # Prepare the plotter class
    pltr = plotter.Plotter(exp_id, boundaries[0], boundaries[1], n_steps, True, obj_fun_type == 'dynamic',
                           plot_final_belief, plot_color_traj, plot_acc_rew, plot_belief_evo)

    # Give the plotter the agent's starting position and first observation
    pltr.add_agent_start(pose, samp_0)
    if plot_belief_evo:
        pltr.gather_belief_data()

    # Start the simulation
    print 'Step:',
    for i in range(n_steps):
        print '{},'.format(i),
        # Agent chooses the next action
        act = agt.select_action(pose)

        # Send High-res trajectory to plotter
        samp_pos, samp_obs = world.execute_full_action(act, plot_traj_res)
        map(pltr.add_agent_step, samp_pos, samp_obs)

        # Execute action and collect observations along the trajectory.
        samp_pos, samp_obs = world.execute_full_action(act)
        pose = world.pose.clone()
        pltr.add_agent_pose(pose)

        # Give the agent a new observation
        rew = acc_rew[-1]
        for p, o in zip(samp_pos, samp_obs):
            agt.observe(p, o)
            rew += o

        acc_rew.append(rew)
        if int(math.log(1 + 1.0 * i)) != int(math.log(1 + 1.0 * (i + 1))):
            print 'Optimizing hyperparameters...',
            agt.optimize()
    print 'Done.'

    # Let's now do some plotting
    pltr.set_acc_reward(acc_rew)
    pltr.set_agent_beleif(agt.belief)
    pltr.display()

    return acc_rew


if __name__ == '__main__':
    # Experiment configuration:
    # 0  experience_id
    # 1  number of episodes
    # 2  Algorithm name (random, MTCS_cont, MTCS_disc, MKCF_FTS_cont)
    # 3  Algorithm arguments:
    #       random: None
    #       MTCS_cont: MCTS Depth, MCTS Iterations, k_mc
    #       MTCS_disc: MCTS Depth, MCTS Iterations, k_mc, use KernelTrajectories
    #       MKCF_FTS_cont: FTS Depth, number of random rollouts
    # 4  UCB exploration exploitation parameter
    # 5  Objective function type (static, dynamic)
    # 6  Number of runs per experiment, to be averaged

    # Number of episodes to run a simulation for
    nb_ep = 20

    # Number of runs per experiment (to average over several runs)
    run_per_exp = 1

    # Plotting arguments
    plot_color_traj = True
    plot_acc_rew = True
    plot_final_belief = True
    plot_belief_evo = False
    plot_all_exp_rewards = True
    plot_traj_res = 16

    # Whether to store accumulated reward in files
    log_in_file = False

    # Define experiments here
    exploration_param = 20.0
    experiments = [
        # (0, nb_ep, 'random', (), exploration_param, 'static', run_per_exp),

        # (1, nb_ep, 'MCTS_disc', (3, 150, 1), exploration_param, 'static', run_per_exp),
        # (2, nb_ep, 'MCTS_disc', (4, 300, 1), exploration_param, 'static', run_per_exp),
        # (3, nb_ep, 'MCTS_disc', (5, 500, 1), exploration_param, 'static', run_per_exp),

        # (4, nb_ep, 'MCTS_cont', (1, 37, 1, True), exploration_param, 'static', run_per_exp),
        # (5, nb_ep, 'MCTS_cont', (2, 75, 1, True), exploration_param, 'static', run_per_exp),
        (6, nb_ep, 'MCTS_cont', (3, 150, 1, True), exploration_param, 'static', run_per_exp),
        # (7, nb_ep, 'MCTS_cont', (4, 300, 1, True), exploration_param, 'static', run_per_exp),
        # (8, nb_ep, 'MCTS_cont', (5, 500, 1, True), exploration_param, 'static', run_per_exp),

        # (9, nb_ep, 'FTS', (1,), exploration_param, 'static', 1),
        # (10, nb_ep, 'FTS', (2,), exploration_param, 'static', 1),
        # (11, nb_ep, 'FTS', (3,), exploration_param, 'static', 1),

        # (12, nb_ep, 'MKCF_FTS_cont', (3, 0), exploration_param, 'static', run_per_exp),
        # (13, nb_ep, 'MKCF_FTS_cont', (4, 1, 0), exploration_param, 'static', run_per_exp),

        # (14, nb_ep, 'MKCF_FTS_cont*', (3, 10), exploration_param, 'static', run_per_exp)
    ]

    # experiments = [(10, nb_ep, 'MKCF_FTS_cont', (3, 0), exploration_param, 'static', run_per_exp)]
    # experiments = [(9, nb_ep, 'FTS', (3,), exploration_param, 'static', 1)]

    # Let's now run all experiments
    all_rewards = {}
    for experiment in experiments:
        experiment_rewards = np.array([])
        tot_time = 0

        # Potentially run an experiment several times
        for run_id in range(experiment[6]):
            exp_id = '{} {}'.format(experiment[2], experiment[3][0] if len(experiment[3]) > 0 else '')
            print 'Starting experiment {} ({} steps), run {} of {}.'.format(exp_id, experiment[1], run_id,
                                                                            experiment[6])
            start_time = time.time()

            # Running an experiment
            rewards = np.array(run_experiment(exp_id, *experiment))

            # Gathering the accumulated rewards for this experiment
            exp_time = time.time() - start_time
            tot_time += exp_time

            # Sleep for plot saving purpuses
            if exp_time < 1.0:
                time.sleep(1.0 - exp_time)

            if len(experiment_rewards) == 0:
                experiment_rewards = np.array([rewards])
            else:
                experiment_rewards = np.append(experiment_rewards, [rewards], axis=0)

        # Averaging time and accumulated rewards across all runs for a single experiement
        tot_time /= float(experiment[6])
        reward_list = list(np.sum(experiment_rewards, axis=0) / float(experiment[6]))
        all_rewards[exp_id] = reward_list
        print 'Experiment', exp_id, 'finished in avg', tot_time, 's,\nwith cumulated rewards:', reward_list

        # Logging results in a file
        if log_in_file:
            f = open('results.txt', 'a+')
            f.write(str(experiment) + '|\t' + str(reward_list) + '\n')
            f.close()

    # Print curv with all rewards
    if plot_all_exp_rewards:
        plt.figure()
        plt.subplot(1, 1, 1)
        for name, rewards in all_rewards.items():
            plt.plot(range(len(rewards)), rewards, label=name)
        plt.legend(loc=2)
        plt.xlabel('Time steps')
        plt.ylabel('Accumulated reward')
        if plotter.save_plot_to_file:
            plt.savefig('res/comparing_rewards' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.png')
        else:
            plt.show()
