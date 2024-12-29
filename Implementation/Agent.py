import numpy as np
import random
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt  # Added for visualization

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, memory_size=2000, target_update_freq=10):
        """
        Initialize Enhanced Deep Q-Learning Agent with Double Q-learning.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.target_update_freq = target_update_freq
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network
        self.update_target_model()  # Initialize target model

    def _build_model(self):
        """
        Build a neural network model.
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Update the target model to match the primary model.
        """
        self.target_model.set_weights(self.model.get_weights())

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
                # Double Q-learning update
                best_action = np.argmax(self.model.predict(next_state)[0])
                target = reward + self.discount_factor * self.target_model.predict(next_state)[0][best_action]
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > 0.01:
            self.exploration_rate *= self.exploration_decay

    def train(self, episodes, env, batch_size):
        """
        Train the agent over multiple episodes in the given environment.
        """
        rewards = []  # Track rewards for each episode
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

            rewards.append(total_reward)  # Save the total reward for this episode

            if episode % self.target_update_freq == 0:
                self.update_target_model()

            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Exploration Rate: {self.exploration_rate:.2f}")

        # Plot the results after training

        self._plot_rewards(rewards)

    def _plot_rewards(self, rewards):
        """
        Plot the rewards over episodes.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label="Total Reward per Episode", color="blue")
        plt.axhline(y=np.mean(rewards), color="red", linestyle="--", label="Average Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Performance")
        plt.legend()
        plt.grid()
        plt.show()

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
