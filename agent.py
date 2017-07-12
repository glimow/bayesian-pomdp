import GPy

from behaviors import disc_MCTS, cont_MCTS
from behaviors.FTS import FTS
from behaviors.cont_KMCF_FTS import ContKmcfFts
from behaviors.randomBehavior import RandomBehavior
from belief import Belief
from environment import world
from environment.Splines2D import Splines2D
from tools import putils

__author__ = 'philippe'

# Action selection BO
act_sel_k = 200  # k (10.0 is a good choice for smooth trajectories)
act_space_res = 12  # resolution on which to find argmax(acquisition function) in BO (8)
epsilon_act_diff = 0.75 ** Splines2D.free_dims  # smallest distance between 2 different actions (=0.01 for 4 dim) 0.316 ** world.World.Spline.free_dims


class Agent:
    def __init__(self, action_provider, init_pose, init_obs, obj_fun_type, exploration_param, behavior_alg,
                 behavior_args):
        """
        Initialize the agent.
        :param action_provider: ActionProvider initialized with the right type of action.
        :param init_pose: Initial pose.
        :param init_obs: Initial observation.
        :param obj_fun_type: Objective function type. Can be 'static' or 'dynamic'
        :param exploration_param: UCB Exploration parameter defining a balance between exploration and exploitation.
        :param behavior_alg: Name of the behavior algorithm to use. Can be 'MCTS_cont', 'MCTS_disc', 'FTS', or 'random'.
        :param behavior_args: Arguments for the behaviour algorithm.
        """
        self.action_provider = action_provider

        # Specify a reward function when simulating actions
        simulator_reward_fun = putils.UCB(exploration_param)

        # Create the agent's belief
        if obj_fun_type == 'static':
            def restrictions(m):
                m['rbf.variance'].constrain_bounded(0.01, 10.0, warning=False)
                m['rbf.lengthscale'].constrain_bounded(0.1, 10.0, warning=False)

            self.belief = Belief(None, GPy.kern.RBF(2), restrict_hyper_parameters=restrictions)
        elif obj_fun_type == 'dynamic':
            # Space-Time kernel
            ker_space = GPy.kern.RBF(2, lengthscale=0.920497128746, variance=0.00133408521113, active_dims=[0, 1])
            ker_time = GPy.kern.PeriodicExponential(lengthscale=25,
                                                    active_dims=[2]) + GPy.kern.Matern52(1, active_dims=[2])

            # Restrictions on hyperparameters when running optimize
            def restrictions(m):
                m['.*periodic_exponential.variance'].constrain_bounded(0.1, 10.0, warning=False)
                m['.*periodic_exponential.period'].constrain_fixed(world.dynamic_function_period, warning=False)
                m['.*periodic_exponential.lengthscale'].constrain_bounded(0.0, 2.0, warning=False)
                m['.*rbf.variance'].constrain_bounded(0.1, 10.0, warning=False)
                m['.*rbf.lengthscale'].constrain_bounded(0.5, 1.0, warning=False)
                m['.*Mat52.variance'].constrain_bounded(0.1, 10.0, warning=False)
                m['.*Gaussian_noise.variance'].constrain_bounded(0.0, 0.2, warning=False)  # .0004

            self.belief = Belief(None, ker_space * ker_time, restrict_hyper_parameters=restrictions)
        else:
            raise Exception('Objective function type', obj_fun_type, 'is not valid.')

        # Initialize the belief
        self.belief.update(init_pose, init_obs)

        # Create the agent's behavior
        if behavior_alg == 'MCTS_cont':
            self.behavior = cont_MCTS.ContMCTS(self.action_provider, simulator_reward_fun, behavior_args[0],
                                               behavior_args[1], behavior_args[2], act_sel_k, epsilon_act_diff)
        elif behavior_alg == 'MCTS_disc':
            self.behavior = disc_MCTS.DiscMCTS(self.action_provider, simulator_reward_fun, behavior_args[0],
                                               behavior_args[1],
                                               behavior_args[2])
        elif behavior_alg == 'FTS':
            self.behavior = FTS(self.action_provider, simulator_reward_fun, behavior_args[0])
        elif behavior_alg == 'MKCF_FTS_cont' or behavior_alg == 'MKCF_FTS_cont*':
            self.behavior = ContKmcfFts(self.action_provider, simulator_reward_fun, behavior_args[0], act_sel_k,
                                        behavior_args[1])
        elif behavior_alg == 'random':
            self.behavior = RandomBehavior()
        else:
            raise Exception('Behavior algorithm', behavior_alg, 'is not valid.')

    def select_action(self, pose):
        """
        Runs the agent's behaviour to select and return an action to execute at the given pose.
        :param pose: Pose from which to plan.
        :return: Action given by the ActionProvider
        """
        return self.behavior.run(pose, self.belief)

    def observe(self, pose, obs):
        """
        Observe the given (pose, observation) pair and update the agent's belief.
        :param pose: Pose
        :param obs: Observation (real number)
        """
        self.belief.update(pose, obs)

    def optimize(self, num_iterations=20):
        """
        Run optimiaztion routines on the agent's belief
        :param num_iterations: number of iterations in the optimization routine (default: 20)
        """
        self.belief.optimize(num_iterations)
