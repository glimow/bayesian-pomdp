import numpy as np
from behaviors.FTS import FTS
from BO import BO
from tools.GM_gradient_descent import gm_gradient_descent
from tools.KMCF import KMCF


class ContKmcfFts(FTS):
    """
    Performs a Trimmed Full Tree Search to solve a POMDP with continuous actions.
    Only the few mode branches of a node are searched.
    """

    def __init__(self, action_provider, simulator_reward_fun, fts_depth, act_sel_k, nb_random_rollouts,
                 max_actions_per_node=10):
        """
        Initializes ContKmfcFts.
        :param action_provider: corresponding ActionProvider.
        :param simulator_reward_fun: reward function.
        :param fts_depth: maximum depth for FTS.
        :param act_sel_k: UCB parameter for the action selection BO.
        :param nb_random_rollouts: Number of random rollouts to compute an action's reward.
        :param max_actions_per_node: convergence parameter for action selection: maximum number of actions per node.
        """
        FTS.__init__(self, action_provider, simulator_reward_fun, fts_depth)
        self.act_sel_k = act_sel_k
        self.max_actions_per_node = max_actions_per_node
        self.nb_random_rollouts = nb_random_rollouts

        if nb_random_rollouts > 0:  # Seems to work better...
            self.act_sel_k *= 10

    def __get_untried_act__(self, node):
        """
        Get an untried action for the given Node.
        :param node: a Node to get an untried action from.
        :return: an untried action for the given Node. Returns None if there are no untried actions.
        """
        # Initialize BO for every node
        if not hasattr(node, 'nb_actions_tried'):
            node.nb_actions_tried = 0
            node.action_picker = BO(self.action_provider.nparams, self.act_sel_k, [-1.0, 1.0], opt_maxeval=100,
                                    steps_btw_opt=10, rbf_lengthscale=3.0, rbf_variance=10.0, custom_gp=True)
            node.nb_actions_tried = 0
            node.applied_KMCF = False

        if node.nb_actions_tried >= self.max_actions_per_node:
            return None
        else:
            node.nb_actions_tried += 1
            return self.action_provider.build_action_from_params([node.action_picker.next_sample()], node.pose)

    def __get_random_act__(self, pose):
        """
        Returns a random action starting from the given pose.
        :param pose: Starting pose for the action.
        :return: A random action.
        """
        params = np.array([np.random.uniform(-1.0, 1.0) for _ in range(self.action_provider.nparams)])
        return self.action_provider.build_action_from_params(params, pose)

    def explore_child(self, from_v):
        """
        Recursive function to consider all actions for a node, trim them, and then explore children.
        :param from_v: start node.
        :return: the accumulated average reward from this node and all its children.
        """
        dont_explore = False
        if self.depth(from_v) == self.fts_depth:
            dont_explore = True
        number_actions = 0
        tries_actions = []
        tries_rewards = []

        # Explore all actions for nore
        untried_a = self.__get_untried_act__(from_v)
        print(untried_a)
        while untried_a is not None:
            number_actions += 1

            # Execute an untried action
            samp_poses = self.action_provider.sample_trajectory(untried_a)
            new_pose = self.action_provider.sample_trajectory_at(untried_a, 1)
            bound_disc = self.action_provider.get_boundary_discount(samp_poses)

            # For each sub position, estimate the reward based on belief mean, variance and boundary discount factor
            new_belief = None
            if self.update_belief_each_step:
                new_belief = from_v.belief.clone()
                rs = 0
                for i, p in enumerate(samp_poses):
                    est_mean, est_var = new_belief.estimate(p)
                    rs += self.simulator_reward_fun(est_mean, bound_disc[i] * est_var)
                    new_belief.update(p, est_mean[0])
                est_mean, est_var = from_v.belief.estimate(samp_poses)
            else:
                est_mean, est_var = from_v.belief.estimate(samp_poses)
                rs = sum(self.simulator_reward_fun(sum(est_mean), sum(bound_disc * est_var)))

            if self.nb_random_rollouts != 0:
                if not new_belief:
                    new_belief = from_v.belief.clone()
                    new_belief.update_all(samp_poses, est_mean[0])
                rs += self.__execute_random_rollouts__(new_belief, new_pose, self.depth(from_v))

            # Keep track of all actions tried and their reward
            tries_actions.append(untried_a)
            tries_rewards.append(rs)

            # Update the action picker BO
            from_v.action_picker.update(self.action_provider.get_low_dim(untried_a), rs)

            # Try another action
            untried_a = self.__get_untried_act__(from_v)

        v_rew = sum(tries_rewards) / float(number_actions)
        if dont_explore:
            return v_rew

        # Only keep the relevant modes with KMCF
        kmcf = KMCF()
        kmcf.param_ker_o = 250
        actions_tries = np.array(map(self.action_provider.get_low_dim, tries_actions))
        # Normalize KMCF input
        # TODO: Don't normalize with different factors for every dim
        maxis = [max(actions_tries[:, i]) for i in range(self.action_provider.nparams)]
        minis = [min(actions_tries[:, i]) for i in range(self.action_provider.nparams)]
        mini = min(minis)
        maxi = max(maxis)
        # actions_tries_nrm = np.array(
        #    [(actions_tries[:, i] - minis[i]) / (maxis[i] - minis[i]) for i in range(self.action_provider.nparams)]).T
        actions_tries_nrm = (actions_tries - mini) / (maxi - mini)
        rewards = np.array(tries_rewards)
        kmcf.set_anchor_pts(actions_tries_nrm, rewards)
        test_pt = np.array([max(rewards)])

        # Get a mixture of gaussian representation of the action space
        mixt_w, mixt_ker = kmcf.test_mix_gaussian(test_pt)

        # Find the modes
        var = 0.01
        covs = np.array([np.identity(self.action_provider.nparams) * var for _ in range(number_actions)])
        modes, modes_vals = gm_gradient_descent(mixt_w.reshape(-1), actions_tries_nrm, covs,
                                                self.action_provider.nparams, 100)
        # Only keep strong modes
        threshold_val = 0.6 * max(modes_vals)
        strong_modes = np.array([m for m, v in zip(modes, modes_vals) if v > threshold_val])

        # Viz
        # import matplotlib.pyplot as plt
        # res = 50
        # # axis_x = [min(actions_tries_nrm[:, 0]), max(actions_tries_nrm[:, 0])]
        # # axis_y = [min(actions_tries_nrm[:, 1]), max(actions_tries_nrm[:, 1])]
        # axis_x = [0, 1]
        # axis_y = [0, 1]
        # space_x = np.linspace(axis_x[0], axis_x[1], res)
        # space_y = np.linspace(axis_y[0], axis_y[1], res)
        # xv, yv = np.meshgrid(space_x, space_y)
        # space_xy = np.array([xv.reshape(-1), yv.reshape(-1)]).T
        # embed = np.dot(mixt_ker(space_xy), mixt_w.T).reshape(((res, res)))
        #
        # plt.figure(figsize=(14, 10))
        # # plt.subplot(1, 2, 1)
        # plt.contourf(space_x, space_y, embed, 16, alpha=.75, cmap='jet')
        # plt.plot(actions_tries_nrm[:, 0], actions_tries_nrm[:, 1], 'g*')
        # plt.plot(modes[:, 0], modes[:, 1], 'g+')
        # plt.plot(strong_modes[:, 0], strong_modes[:, 1], 'ro')
        #
        # # node_bo = from_v.action_picker
        # # space_x = np.linspace(node_bo.search_min, node_bo.search_max, res)
        # # space_y = np.linspace(node_bo.search_min, node_bo.search_max, res)
        # # xv, yv = np.meshgrid(space_x, space_y)
        # # space_xy = np.array([xv.reshape(-1), yv.reshape(-1)]).T
        # # gp_mean, gp_var = node_bo.gp.estimate(space_xy)
        # # plt.subplot(1, 2, 2)
        # # plt.contourf(space_x, space_y, gp_mean.reshape(res, res), 16, alpha=.75, cmap='jet')
        # # plt.plot(node_bo.x[:, 0], node_bo.x[:, 1], 'g*')
        # plt.show()

        # Only keep the actions that are on the modes and explore deeper levels for those actions
        for act_params in strong_modes:
            # Execute the mode actions
            action = self.action_provider.build_action_from_params(act_params, from_v.pose)
            samp_poses = self.action_provider.sample_trajectory(action)
            new_pose = self.action_provider.sample_trajectory_at(action, 1)
            bound_disc = self.action_provider.get_boundary_discount(samp_poses)

            # For each sub position, estimate the reward based on belief mean, variance and boundary discount factor
            new_belief = None
            if self.update_belief_each_step:
                new_belief = from_v.belief.clone()
                rs = 0
                for i, p in enumerate(samp_poses):
                    est_mean, est_var = new_belief.estimate(p)
                    rs += self.simulator_reward_fun(est_mean, bound_disc[i] * est_var)
                    new_belief.update(p, est_mean[0])
                est_mean, est_var = from_v.belief.estimate(samp_poses)
            else:
                est_mean, est_var = from_v.belief.estimate(samp_poses)
                rs = sum(self.simulator_reward_fun(sum(est_mean), sum(bound_disc * est_var)))

            if self.nb_random_rollouts != 0:
                if not new_belief:
                    new_belief = from_v.belief.clone()
                    new_belief.update_all(samp_poses, est_mean[0])
                rs += self.__execute_random_rollouts__(new_belief, new_pose, self.depth(from_v))

            new_v = FTS.Node(new_pose, rs, from_v, action, new_belief, None)
            v_rew += self.explore_child(new_v) / len(strong_modes)

        # Propagate accumulated reward up the tree
        return v_rew

    def __execute_random_rollouts__(self, belief, pose, curr_depth):
        """
        Executes random sequences of actions starting from the current pose/belief. Computes the average accumulated
        reward of taking these random actions.
        :param belief: Current belief.
        :param pose: Current pose.
        :param curr_depth: Node depth.
        :return: Average accumulated reward.
        """
        r = 0
        for i in range(self.nb_random_rollouts):
            d = curr_depth
            p = pose
            while d < self.fts_depth:
                # Pick a random action
                act = self.__get_random_act__(p)

                # Execute the action
                samp_poses = self.action_provider.sample_trajectory(act)
                new_p = self.action_provider.sample_trajectory_at(act, 1)
                bound_disc = self.action_provider.get_boundary_discount(samp_poses)

                # Estimate accumulated reward for action
                est_mean, est_var = belief.estimate(samp_poses)
                r += self.simulator_reward_fun(sum(est_mean), sum(bound_disc * est_var))

                p = new_p
                d += 1
        r /= float(self.nb_random_rollouts)
        return r
