from qlearning import QLearning

sentence = "Q-Learning VS DQN In Letter Order Prediction"
episodes = 3000
epsilon = 0.1       # probability of random actions
alpha = 0.1       # learning rate
gamma = 0.9       # importance of future rewards
print_interval = 100

qlearning = QLearning(sentence, episodes, epsilon, alpha, gamma, print_interval)
qlearning.train()
