import math

__author__ = 'philippe'
reward_min = 0.00001


class MCTS:
    """
    Performs a Monte-Carlo Tree Search to solve a POMDP. Abstract class.
    """

    def __init__(self, action_provider, simulator_reward_fun, mcts_depth, mcts_max_iter, k_mc):
        """
        Initializes MCTS.
        :param action_provider: corresponding ActionProvider.
        :param simulator_reward_fun: reward function.
        :param mcts_depth: maximum depth for MCTS.
        :param mcts_max_iter: maximum number of iterations from MCTS.
        :param k_mc: MCTS exploration-exploitation parameter.
        """
        self.mcts_depth = mcts_depth  # 5 sounds good
        self.mcts_max_iter = mcts_max_iter  # 20 is fast enough
        self.simulator_reward_fun = simulator_reward_fun
        self.update_belief_each_step = True
        self.action_provider = action_provider
        self.k_mc = k_mc

    def run(self, from_pose, belief):
        """
        Run the MCTS algorithm.
        :param from_pose: agent starting Pose.
        :param belief: agent's belief.
        :return: the next action to execute.
        """
        v0 = self.new_node(belief.clone(), from_pose, reward_min, None, None)
        for i in range(self.mcts_max_iter):
            vi = self.tree_policy(v0)
            r = self.default_policy(vi)
            self.__backup__(vi, r)
        return self.best_child(v0).from_act

    def tree_policy(self, v0):
        """
        MCTS tree exploration policy.
        :param v0: Node to start the exploration from.
        :return: the best child Node with depth=mcts_depth
        """
        v = v0
        while self.depth(v) <= self.mcts_depth:
            untried_a = self.get_untried_act(v)
            if not untried_a:
                v = self.best_child(v)
            else:
                # Sample positions along the trajectory
                samp_poses = self.action_provider.sample_trajectory(untried_a)
                new_belief = v.belief.clone()

                if self.update_belief_each_step:
                    # For each sub position, estimate a reward and update the belief
                    rs = 0
                    bound_disc = self.action_provider.get_boundary_discount(samp_poses)
                    for i, p in enumerate(samp_poses):
                        est_mean, est_var = new_belief.estimate(p)
                        rs += self.simulator_reward_fun(est_mean[0], bound_disc[i] * est_var[0])
                        new_belief.update(p, est_mean[0])
                else:
                    est_mean, est_var = new_belief.estimate(samp_poses)
                    new_belief.update_all(samp_poses, est_mean)
                    bound_disc = self.action_provider.get_boundary_discount(samp_poses)
                    rs = self.simulator_reward_fun(sum(est_mean), sum(bound_disc * est_var))

                # Update the action picker BO
                self.update_untried_act(v, untried_a, rs)

                # Create a new node
                new_pose = self.action_provider.sample_trajectory_at(untried_a, 1)
                return self.new_node(new_belief, new_pose, rs, v, untried_a)
        return v

    def default_policy(self, v):
        """
        Random tree exploration policy.
        :param v: Node to start the exploration from.
        :return: the accumulated reward when executing random actions starting from v.
        """
        r = self.get_acc_reward(v)
        new_pos = v.pose.clone()
        d = self.depth(v)
        belief = v.belief.clone()
        while d <= self.mcts_depth:
            # Pick a random action
            a = self.get_random_act(new_pos)

            # Sample positions along the trajectory
            samp_poses = self.action_provider.sample_trajectory(a)

            if self.update_belief_each_step:
                # For each sub position, estimate a reward and update the belief
                rs = 0
                bound_disc = self.action_provider.get_boundary_discount(samp_poses)
                for i, p in enumerate(samp_poses):
                    est_mean, est_var = belief.estimate(p)
                    rs += self.simulator_reward_fun(est_mean[0], bound_disc[i] * est_var[0])
                    belief.update(p, est_mean[0])
            else:
                est_mean, est_var = belief.estimate(samp_poses)
                belief.update_all(samp_poses, est_mean)
                bound_disc = self.action_provider.get_boundary_discount(samp_poses)
                rs = self.simulator_reward_fun(sum(est_mean), sum(bound_disc * est_var))

            # Update the new position
            new_pos = self.action_provider.sample_trajectory_at(a, 1)
            r += rs
            d += 1
        return r

    @staticmethod
    def __backup__(vi, r):
        """
        Propagate up the tree the number of time a node's children are visited and their accumulated reward.
        :param vi: Node to start from.
        :param r: reward to propagate up the tree.
        """
        v = vi
        while v.parent is not None:
            v.visited += 1.0
            v.reward += r
            v = v.parent

    class Node:
        """
        Node class for Tree searches.
        """

        def __init__(self):
            self.belief = None
            self.pose = None
            self.visited = 0.0
            self.reward = 0.0
            self.children = []
            self.from_act = None
            self.parent = None

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
        node = self.Node()
        node.belief = belief
        node.pose = pose
        node.visited = 1.0
        node.reward = reward
        node.children = []
        node.from_act = from_act
        node.parent = parent
        if parent is not None:
            parent.children.append(node)
        return node

    def get_random_act(self, new_pose):
        """
        Returns a random action.
        :param new_pose: starting agent Pose.
        :return: random action.
        """
        raise NotImplementedError("Implement this method")

    def get_untried_act(self, node):
        """
        Returns an untried action for this Node.
        :param node: Node from which to get an untried action.
        :return: an untried action.
        """
        raise NotImplementedError("Implement this method")

    def update_untried_act(self, node, act, rew):
        """
        Updates an untried action with its corresponding reward.
        :param node: the Node to be updated.
        :param act: the action to update.
        :param rew: the corresponding reward.
        """
        raise NotImplementedError("Implement this method")

    @staticmethod
    def get_acc_reward(node):
        """
        Returns the accumulated reward of a node in the tree.
        :param node: a Node.
        :return: the node's accumulated reward.
        """
        reward = node.reward
        while node.parent is not None:
            node = node.parent
            reward += node.reward
        return reward

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

    def best_child(self, vp):
        """
        Returns the best child according to a metric.
        :param vp: parent node.
        :return: best child.
        """
        g = {}
        n_p = vp.visited
        for vi in vp.children:
            n_i = vi.visited
            r_i = self.get_acc_reward(vi)
            g[vi] = self.k_mc * math.sqrt(2 * math.log(n_p) / n_i) + r_i / n_i
            # g[vi] = self.get_acc_reward(vi)

        best = None
        best_g = float('-inf')
        for child, score in g.iteritems():
            if score > best_g:
                best_g = score
                best = child
        return best
