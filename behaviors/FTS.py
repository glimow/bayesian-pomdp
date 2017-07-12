import random

from environment.Splines2D import DiscreteSplines2D

__author__ = 'philippe'
reward_min = 0.00001


class FTS:
    """
    Performs a Full Tree Search to solve a POMDP.
    """

    def __init__(self, action_provider, simulator_reward_fun, fts_depth=3):
        """
        Initializes FTS.
        :param action_provider: corresponding ActionProvider.
        :param simulator_reward_fun: simulates the reward function.
        :param fts_depth: maximum depth for the full-tree search (default 3)
        """
        self.action_provider = action_provider
        self.fts_depth = fts_depth
        self.simulator_reward_fun = simulator_reward_fun
        self.update_belief_each_step = True

    def run(self, from_pose, belief):
        """
        Run the FTS algorithm.
        :param from_pose: agent starting Pose
        :param belief: agent's belief
        :return: the next action to execute
        """
        v0 = FTS.Node(from_pose, 0, None, None, belief.clone(), DiscreteSplines2D.get_splines(from_pose))
        self.explore_child(v0)
        return self.__best_child__(v0).from_act

    def explore_child(self, from_v):
        """
        Recursive function to explore the next level of the tree.
        :param from_v: praent node.
        :return: the accumulated average reward from this node and all its children.
        """
        dont_explore = False
        if self.depth(from_v) == self.fts_depth:
            dont_explore = True
        v_rew = 0
        number_actions = 0
        untried_a = self.__get_untried_act__(from_v)
        while untried_a is not None:
            number_actions += 1
            # Execute action
            samp_poses = self.action_provider.sample_trajectory(untried_a)
            new_pose = self.action_provider.sample_trajectory_at(untried_a, 1)
            bound_disc = self.action_provider.get_boundary_discount(samp_poses)

            if self.update_belief_each_step:
                new_belief = from_v.belief.clone()
                rs = 0
                for i, p in enumerate(samp_poses):
                    est_mean, est_var = new_belief.estimate(p)
                    rs += self.simulator_reward_fun(est_mean, bound_disc[i] * est_var)
                    new_belief.update(p, est_mean[0])
            else:
                est_mean, est_var = from_v.belief.estimate(samp_poses)
                rs = self.simulator_reward_fun(sum(est_mean), sum(bound_disc * est_var))
                if not dont_explore:
                    new_belief = from_v.belief.clone()
                    new_belief.update_all(samp_poses, est_mean)

            # Explore child
            if dont_explore:
                v_rew += rs
            else:
                new_v = FTS.Node(new_pose, rs, from_v, untried_a, new_belief, DiscreteSplines2D.get_splines(new_pose))
                v_rew += self.explore_child(new_v)

            # Try another action
            untried_a = self.__get_untried_act__(from_v)
        from_v.reward += v_rew / float(number_actions)
        return from_v.reward

    def __get_untried_act__(self, node):
        """
        Get an untried action for the given Node.
        :param node: a Node to get an untried action from.
        :return: an untried action for the given Node. Returns None if there are no untried actions.
        """
        if len(node.untried_acts) == 0:
            return None
        else:
            return node.untried_acts.pop(random.randint(0, len(node.untried_acts) - 1))

    @staticmethod
    def __best_child__(vp):
        """
        Returns the child with the best accumulated reward.
        :param vp: parent node.
        :return: child with maximum accumulated reward
        """
        g = {}
        for vi in vp.children:
            g[vi] = vi.reward

        best = None
        best_g = float('-inf')
        for child, score in g.iteritems():
            if score > best_g:
                best_g = score
                best = child
        return best

    class Node:
        """
        Node class helper for FTS.
        """

        def __init__(self, pose, reward, parent, from_act, belief, actions):
            """
            Initializes a Node.
            :param pose: node Pose.
            :param reward: node reward (from action that led to this node).
            :param parent: parent Node.
            :param from_act: action that led to this node.
            :param belief: belief for the current Node.
            :param actions: list of actions that can be taken from this Node.
            """
            self.pose = pose
            self.reward = reward
            self.belief = belief
            self.children = []
            self.from_act = from_act
            self.untried_acts = actions
            self.parent = parent
            if parent is not None:
                parent.children.append(self)

    @staticmethod
    def depth(node):
        """
        Returns the depth of the node in the tree.
        :param node: a Node.
        :return: the node's depth in the tree.
        """
        d = 0
        while node.parent is not None:
            node = node.parent
            d += 1
        return d
