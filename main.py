import gym
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")

# paramaters
batch_size = 32
n_episodes = 1001

# agent
class DQNAgent:
    def __init__(self):
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.discount_factor = 0.95
        self.learning_rate = 0.001

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))  # Input layer
        model.add(Dense(24, activation='relu'))               # Hidden layer
        model.add(Dense(2, activation='linear'))              # Output layer
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if self.epsilon > random.random():
            action = env.action_space.sample()
        else: 
            action = np.argmax(self.model.predict(state.reshape(1, -1), verbose=0))
        return action
    
    def remember(self, state, action, reward, done, new_state):
        self.memory.append((state, action, reward, done, new_state))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, done, new_state in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(new_state.reshape(1, -1), verbose=0)[0])) 
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQNAgent()  # initialize agent

# TRAINING
for i in range(n_episodes):

    state, info = env.reset()  
    env.render()
    score = 0
    done = False  
    
    while not done:
        env.render()

        # pick action
        action = agent.act(state)  

        # execute action
        new_state, reward, done, _, _ = env.step(action)

        # store in replay memory
        reward = reward if not done else -10
        agent.remember(state, action, reward, done, new_state)

        # update state
        state = new_state

        # add 1 to score
        score += 1

    # print info about episode
    print(f"episode: {i+1}, score: {score}, epsilon: {agent.epsilon}")
    
    # learn from replay memory
    agent.replay(batch_size)

    # reset score
    score = 0

# save model
agent.model.save("cartpoledqn.h5")
