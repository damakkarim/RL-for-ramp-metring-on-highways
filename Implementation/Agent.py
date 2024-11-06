import numpy as np
import random
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, memory_size=2000):
        """
        Initialize Deep Q-Learning Agent.
        :param state_size: The size of the state space.
        :param action_size: The size of the action space.
        :param learning_rate: Learning rate for the neural network.
        :param discount_factor: Discount factor for future rewards.
        :param exploration_rate: Initial exploration rate.
        :param exploration_decay: Decay rate for the exploration.
        :param memory_size: Size of the memory buffer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.model = self._build_model()

    def _build_model(self):
        """
        Build a neural network model.
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        Choose the action based on the current state.
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Train the model using randomly sampled experiences from memory.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > 0.01:
            self.exploration_rate *= self.exploration_decay

    def save(self, filename):
        """
        Save the trained model.
        """
        self.model.save(filename)

    def load(self, filename):
        """
        Load a trained model.
        """
        self.model.load_weights(filename)