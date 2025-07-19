import gym
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable) # ajanin beyni

episodes = 1000 # episode
alpha = 0.5 # learning rate
gamma = 0.9 # discount rate

outcomes = []	

# training
for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False
    outcomes.append("Failure")
    
    while not done:
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
        
        # update q table
        qtable[state, action] =  qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        state = new_state
        
        if reward:
            outcomes[-1] = "Success"
        
print("Qtable After Training: ")  
print(qtable)

plt.bar(range(episodes), outcomes)
plt.show()