import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class DQN(nn.Module):
    def __init__(self, sentence, episodes, epsilon, epsilon_decay, min_epsilon, gamma, max_storage, batch_size, print_interval):
        super().__init__()
        self.letters = list(sentence)
        sorted_letters = sorted(set(self.letters))
        self.num_actions = len(sorted_letters)

        action_to_index = {char: idx for idx, char in enumerate(sorted_letters)}
        self.index_to_action = {idx: char for char, idx in action_to_index.items()}
        self.num_states = len(self.letters)

        self.model = QNetwork(self.num_states, self.num_actions)
        self.target_model = QNetwork(self.num_states, self.num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters())

        self.episodes = episodes
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=max_storage)
        self.batch_size = batch_size
        self.print_interval = print_interval

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        epsilon_value = self.epsilon
        best_reward = -float('inf')
        best_sequence = []

        for episode in range(self.episodes):
            state = 0
            done = False
            reward_total = 0
            action_sequence = []

            while not done:
                state_vector = [1 if i == state else 0 for i in range(self.num_states)]

                action = self.select_action(state_vector, epsilon_value)
                action_char = self.index_to_action[action]
                action_sequence.append(action_char)
                correct_action = self.letters[state]

                if action_char == correct_action:
                    reward = 1
                    next_state = state + 1
                    done = next_state == self.num_states
                else:
                    reward = -1
                    next_state = state
                    done = True

                reward_total += reward

                next_state_vector = [1 if i == next_state else 0 for i in range(self.num_states)]
                self.replay_buffer.append((state_vector, action, reward, next_state_vector, done))
                self.optimize_model()

                state = next_state

            if reward_total > best_reward:
                best_reward = reward_total
                best_sequence = action_sequence

            if (episode + 1) % self.print_interval == 0:
                print(f"Episode {episode + 1}, Total Reward: {reward_total}, Epsilon: {epsilon_value}")
                epsilon_value = max(self.min_epsilon, epsilon_value * self.epsilon_decay)

            if (episode + 1) % self.print_interval == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        print("\nBest reward:", best_reward)
        print("Letters from the best reward episode:", "".join(best_sequence))
