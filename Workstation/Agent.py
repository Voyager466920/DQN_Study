import torch
import gymnasium
from Workstation.DQN import DQN
from Experiment_Replay import ReplayMemory

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self,
                 gymnasium_id,
                 replay_memory_size,
                 mini_batch_size,
                 epsilon_init,
                 epsilon_decay,
                 epsilon_min,
                 network_sync_rate,
                 learning_rate_a,
                 discount_factor_g,
                 enable_double_dqn=False):

        self.gymnasium_id = gymnasium_id
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.network_sync_rate = network_sync_rate
        self.learning_rate_a = learning_rate_a
        self.discount_factor_g = discount_factor_g
        self.double_dqn = enable_double_dqn

        self.env = None
        self.policy_dqn = None
        self.target_dqn = None
        self.optimizer = None
        self.memory = None

    def setup(self, render=False):
        self.env = gymnasium.make(self.gymnasium_id, render_mode="human" if render else None)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.policy_dqn = DQN(num_states, num_actions).to(device)
        self.target_dqn = DQN(num_states, num_actions).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)
        self.memory = ReplayMemory(self.replay_memory_size)

        return self.env, self.policy_dqn, self.target_dqn, self.optimizer, self.memory
