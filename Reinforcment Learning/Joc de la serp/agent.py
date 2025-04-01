import torch
import random
import numpy as np
import torch.nn as nn
from model import Network
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from tensordict import TensorDict
from model import Network, QTrainer
from snake_game import Point, SnakeGame, Direction
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

LR = 0.001
GAMMA               = 0.9
EPSILON             = 1
EPSILON_DECAY       = 0.95
EPSILON_MIN         = 0.01
BATCH_SIZE          = 64
SYNC_NETWORK_RATE   = 1000
MEMORY_LENGTH       = 100000
SAMPLE_SIZE         = 1000

class Agent:
    def __init__(self,
                 lr=LR,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 epsilon_decay=EPSILON_DECAY,
                 epsilon_min=EPSILON_MIN,
                 sync_network_rate=SYNC_NETWORK_RATE,
                 batch_size=BATCH_SIZE):

        self.game_counter = 0
        self.learn_step_counter = 0
        self.gamma = gamma # Discount rate
        self.lr = lr
        self.online_model = Network(11, 256, 3)
        self.target_model = Network(11, 256, 3, freeze=True)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sync_network_rate = sync_network_rate
        self.batch_size = batch_size
 
        storage = LazyMemmapStorage(MEMORY_LENGTH)
        self.memory = TensorDictReplayBuffer(storage=storage)

        self.optimizer = optim.Adam(self.online_model.parameters(), lr)
        self.criterion = nn.MSELoss()

    def state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - game.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - game.BLOCK_SIZE)
        point_r = Point(head.x + game.BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + game.BLOCK_SIZE)

        # Distància en format decimal fins la paret més propera
        dist_wall_left = head.x / game.w
        dist_wall_right = (game.w - head.x) / game.w
        dist_wall_up = head.y / game.h
        dist_wall_down = (game.h - head.y) / game.h

        # Distància relativa a la poma
        dist_food_x = (game.food.x - head.x) / game.w
        dist_food_y = (game.food.y - head.y) / game.h
        
        # Les variables tindran valor 1 si es mouen en la seva direcció, sino serà 0
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN

        # Les variables tindran valor 1 si hi ha perill de morir girant en la seva direcció
        danger_straight = game.is_collision(Point(head.x + (dir_r - dir_l) * game.BLOCK_SIZE, head.y + (dir_d - dir_u) * game.BLOCK_SIZE))
        danger_left = game.is_collision(Point(head.x + (dir_u - dir_d) * game.BLOCK_SIZE, head.y + (dir_l - dir_r) * game.BLOCK_SIZE))
        danger_right = game.is_collision(Point(head.x + (dir_d - dir_u) * game.BLOCK_SIZE, head.y + (dir_r - dir_l) * game.BLOCK_SIZE))

        state = [
            dist_wall_left, dist_wall_right, dist_wall_up, dist_wall_down,
            dist_food_x, dist_food_y,
            dir_l, dir_u, dir_r, dir_d,
            danger_straight, danger_left, danger_right
        ]

        return np.array(state, dtype=int)
    
    def action(self, state):
        if np.random.random() < self.epsilon:
            index = np.random.randint(3)
        else:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float) \
                .unsqueeze(0) \
                .to(self.online_model.device)
            index = self.online_model(state_tensor).argmax().item()
        return index
    
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_model.load_state_dict(self.online_model.state_dict())
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.sync_networks()

        self.optimizer.zero_grad()

        samples = self.memory.sample(self.batch_size).to(self.online_model.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_model(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        target_q_values = self.target_model(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.criterion(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_step(self, state, action, reward, new_state, done):
        self.memory.add(TensorDict({ 
                                    "state": torch.tensor(state, dtype=torch.float),
                                    "action": torch.tensor(np.array(action), dtype=torch.long),
                                    "reward": torch.tensor(reward, dtype=torch.float),
                                    "next_state": torch.tensor(new_state, dtype=torch.float),
                                    "done": torch.tensor(done)
                                }, batch_size=[]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    
def play():
    max_score = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        total_reward = 0
        done = False
        while not done:
            state = agent.state(game)
            action = agent.action(state)
            reward, done, score = game.play_step(action)
            new_state = agent.state(game)

            total_reward += reward

            agent.save_step(state, action, reward, new_state, done)
            agent.train()

        game.reset()
        agent.game_counter += 1

        if score > max_score:
            max_score = score
            agent.online_model.save()

        print('Game: ', agent.game_counter, 'Score: ', score, 'Max Score: ', max_score, 'Total reward: ', total_reward)

if __name__ == '__main__':
    play()