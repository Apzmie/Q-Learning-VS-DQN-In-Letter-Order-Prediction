# Q-Learning-VS-DQN-In-Letter-Order-Prediction
With the rise of neural networks, there have been continuous efforts to apply them to reinforcement learning. Deep Q-Network is an approach that applies neural networks to Q-Learning. Q-learning uses a table to store Q-values, while DQN uses neural networks to approximate them. This project compares two reinforcement learning methods to understand their differences on the simple task without visual rendering, based on mathematical equations of Q-Learning and DQN.

# qlearning_train.py
```python
from qlearning import QLearning

sentence = "Q-Learning VS DQN In Letter Order Prediction"
episodes = 3000
epsilon = 0.1       # probability of random actions
alpha = 0.1       # learning rate
gamma = 0.9       # importance of future rewards
print_interval = 100

qlearning = QLearning(sentence, episodes, epsilon, alpha, gamma, print_interval)
qlearning.train()
```
```text
Episode 2900 finished. Total reward: 17
Episode 3000 finished. Total reward: 5

Best reward: 29
Letters from the best reward episode: Q-Learning VS DQN In Letter OrI
```

# dqn_train.py
```python
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
```
```text
Episode 2900, Total Reward: 1, Epsilon: 0.086905299554526
Episode 3000, Total Reward: 6, Epsilon: 0.08647077305675337

Best reward: 15
Letters from the best reward episode: Q-Learning VS DQQ
```

# Comparison
Common point
- During training, rewards do not gradually increase every epoch but rises at irregular intervals.

Difference
- Q-Learning gets higher rewards than DQN, but DQN is a little bit more stable than Q-Learning, as it shows less randomness in reward values during training.

My opinion
- Table-based Q-Learning can learn faster and more accurately than DQN in simple environments like this task because the small number of states allows for direct updates of Q-values in a table.

# Conclusion
- Q-Learning is more efficient in simple environments, while DQN is more efficient in complex environments.
