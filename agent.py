import random
import copy
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

class DDQNAgent(object):
  """ Deep Double Q-learning agent """
  def __init__(self, state_size, action_size, is_eval=False, model_name=""):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=1000)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model_name = model_name
    self.model = load_model(model_name) if is_eval else self.Qmodel()

  def Qmodel(self):
    model = Sequential()
    model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(self.action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # returns action

  def replay(self, batch_size=32):
    """ vectorized implementation; 30x speed up compared with for loop """
    minibatch = random.sample(self.memory, batch_size)
    states = np.array([tup[0][0] for tup in minibatch])
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3][0] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])

    indices = np.argmax(self.model.predict(states), axis=1)
    maxq = self.model.predict(next_states)
    target = copy.deepcopy(self.model.predict(states))

    for j in range(batch_size):
      target[j, actions[j]] = rewards[j] + self.gamma * maxq[j, indices[j]] * (not done[j])

    self.model.fit(states, target, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
