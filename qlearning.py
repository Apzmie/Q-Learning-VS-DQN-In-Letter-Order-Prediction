import numpy as np
import random

class QLearning:
    def __init__(self, sentence, episodes, epsilon, alpha, gamma, print_interval):
        self.letters = list(sentence)
        sorted_letters = sorted(set(self.letters))
        self.num_actions = len(sorted_letters)

        action_to_index = {char: idx for idx, char in enumerate(sorted_letters)}
        self.index_to_action = {idx: char for char, idx in action_to_index.items()}

        self.num_states = len(self.letters)
        self.Q_table = np.zeros((self.num_states, self.num_actions))

        self.episodes = episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.print_interval = print_interval

    def train(self):
        best_reward = float('-inf')
        best_sequence = []

        for episode in range(self.episodes):
            state = 0
            reward_total = 0
            done = False
            action_sequence = []

            while not done:
                if random.random() < self.epsilon:
                    action_index = random.randint(0, self.num_actions - 1)
                else:
                    action_index = np.argmax(self.Q_table[state])

                action = self.index_to_action[action_index]
                correct_action = self.letters[state]

                action_sequence.append(action)

                if action == correct_action:
                    reward = 1
                    next_state = state + 1
                    done = next_state == self.num_states

                    if not done:
                        self.Q_table[state, action_index] += self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state, action_index])
                        state = next_state
                    else:
                        self.Q_table[state, action_index] += self.alpha * (reward - self.Q_table[state, action_index])
                else:
                    reward = -1
                    self.Q_table[state, action_index] += self.alpha * (reward - self.Q_table[state, action_index])
                    done = True

                reward_total += reward

            if (episode + 1) % self.print_interval == 0:
                if reward_total > best_reward:
                    best_reward = reward_total
                    best_sequence = action_sequence.copy()
                print(f"Episode {episode + 1} finished. Total reward: {reward_total}")

        print("\nBest reward:", best_reward)
        print("Letters from the best reward episode:", "".join(best_sequence))
