import random

from MCTS import MCTS

__author__ = 'philippe'


class DiscMCTS(MCTS):
    """
    Performs a Monte-Carlo Tree Search to solve a POMDP with discrete actions.
    """

    def __init__(self, action_provider, simulator_reward_fun, mcts_depth, mcts_max_iter, k_mc):
        """
        Initializes DiscMCTS.
        :param action_provider: corresponding ActionProvider.
        :param simulator_reward_fun: reward function.
        :param mcts_depth: maximum depth for MCTS.
        :param mcts_max_iter: maximum number of iterations from MCTS.
        :param k_mc: MCTS exploration-exploitation parameter.
        """
        MCTS.__init__(self, action_provider, simulator_reward_fun, mcts_depth, mcts_max_iter, k_mc)
        self.k_mc = k_mc  # tree branching factor

    def get_untried_act(self, node):
        """
        Returns an untried action for this Node.
        :param node: Node from which to get an untried action.
        :return: an untried action.
        """
        if len(node.untried_acts) == 0:
            return None
        else:
            return node.untried_acts.pop(random.randint(0, len(node.untried_acts) - 1))

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
        node.untried_acts = self.action_provider.buid_action_from_params(None, pose)
        return node

    def get_random_act(self, new_pose):
        """
        Returns a random action.
        :param new_pose: starting agent Pose.
        :return: random action.
        """
        acts = self.action_provider.buid_action_from_params(None, new_pose)
        return acts[random.randint(0, len(acts) - 1)]

    def update_untried_act(self, node, act, rew):
        pass
