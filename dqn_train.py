from dqn import DQN

sentence = "Q-Learning VS DQN In Letter Order Prediction"
episodes = 3000
epsilon = 0.1       # probability of random actions
epsilon_decay = 0.995       # decay rate of epsilon for each episode
min_epsilon = 0.01        # minimum epsilon
gamma = 0.9       # importance of future rewards
max_storage = 100       # maximum number of experiences to store in replay buffer
batch_size = 32       # number of experiences to use in updating the model
print_interval = 100

dqn = DQN(sentence, episodes, epsilon, epsilon_decay, min_epsilon, gamma, max_storage, batch_size, print_interval)
dqn.train()
