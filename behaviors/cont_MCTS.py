import random

from BO import BO
from MCTS import MCTS

__author__ = 'philippe'


class ContMCTS(MCTS):
    """
    Performs a Monte-Carlo Tree Search to solve a POMDP with continuous actions.
    """

    def __init__(self, action_provider, simulator_reward_fun, mcts_depth, mcts_max_iter, k_mc, act_sel_k,
                 epsilon_act_diff, max_actions_per_node=30):
        """
        Initializes DiscMCTS.
        :param action_provider: corresponding ActionProvider.
        :param simulator_reward_fun: reward function.
        :param mcts_depth: maximum depth for MCTS.
        :param mcts_max_iter: maximum number of iterations from MCTS.
        :param k_mc: MCTS exploration-exploitation parameter.
        :param act_sel_k: UCB parameter for the action selection BO.
        :param epsilon_act_diff: convergence parameter for action selection: minimum distance between two consecutive
        actions before stopping. (Not in use)
        :param max_actions_per_node: convergence parameter for action selection: maximum number of actions per node.
        """
        MCTS.__init__(self, action_provider, simulator_reward_fun, mcts_depth, mcts_max_iter, k_mc)
        self.act_sel_k = act_sel_k
        self.epsilon_act_diff = epsilon_act_diff
        self.max_actions_per_node = max_actions_per_node

    def get_untried_act(self, node):
        """
        Returns an untried action for this Node.
        :param node: Node from which to get an untried action.
        :return: an untried action.
        """
        if node.nb_actions_tried >= self.max_actions_per_node:
            return None
        else:
            node.nb_actions_tried += 1
            return self.action_provider.buid_action_from_params(node.action_picker.next_sample(), node.pose)

    def get_random_act(self, new_pose):
        """
        Returns a random action.
        :param new_pose: starting agent Pose.
        :return: random action.
        """
        ld_params = [random.uniform(-1.0, 1.0) for _ in range(self.action_provider.nparams)]
        return self.action_provider.buid_action_from_params(ld_params, new_pose)

    def new_node(self, belief, pose, reward, parent, from_act):
        """
        Creates a new Node.
        :param belief: Node's own belief.
        :param pose: resulting Pose for the node.
        :param reward: accumulated reward for the node.
        :param parent: parent of this node.
        :param from_act: action which led to this node.
        :return:
        """
        node = MCTS.new_node(self, belief, pose, reward, parent, from_act)
        node.action_picker = BO(self.action_provider.nparams, self.act_sel_k, [-1.0, 1.0], opt_maxeval=100,
                                steps_btw_opt=10, custom_gp=True)
        node.nb_actions_tried = 0

        if self.action_provider.type == 'cont_spline*':
            # Initialize action picker with 10 discrete splines
            disc_plines = [
                [0, 2, 0, 1],
                [0, 0.5, 0, 1],
                [0, 0, 0, 1],
                [0, -2, 0, 1],
                [0, -0.5, 0, 1],

                [0, 2, 0, -1],
                [0, 0.5, 0, -1],
                [0, 0, 0, -1],
                [0, -2, 0, -1],
                [0, -0.5, 0, -1]
            ]
            for spl in disc_plines:
                act = self.action_provider.buid_action_from_params(spl, node.pose)
                samp_pose = self.action_provider.sample_trajectory(act)
                est = node.belief.estimate(samp_pose)
                rs = self.simulator_reward_fun(sum(est[0]), sum(est[1]))
                node.action_picker.update(spl, rs)
        return node

    def update_untried_act(self, node, act, rew):
        """
        Updates an untried action with its corresponding reward.
        :param node: the Node to be updated.
        :param act: the action to update.
        :param rew: the corresponding reward.
        """
        node.action_picker.update(self.action_provider.get_low_dim(act), rew)
