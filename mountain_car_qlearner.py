#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:31:38 2024

@author: juan
"""

import gymnasium as gym
import numpy as np
import os

MAX_NUM_EPISODES = 5000
STEPS_PER_EPISODE = 200

EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30



class QLearner(object):
    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bins
        
        
        
        self.action_shape = environment.action_space.n
        self.Q = np.zeros((self.obs_bins+1,self.obs_bins+1, self.action_shape))
        #matriz de 31*31*3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0
        

    def discretize(self, obs):
        obs = obs
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int))
        
        
    
    def get_action(self, obs):
        discret_obs = self.discretize(obs)
        
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            #print(discret_obs)
            return np.argmax(self.Q[discret_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])
        
        
    def learn(self, obs, action, reward, next_obs):
        discret_obs = self.discretize(obs)
        discret_next_obs = self.discretize(next_obs)
        
        td_target = reward + self.gamma * np.max(self.Q[discret_next_obs])
        td_error = td_target - self.Q[discret_obs][action]
        self.Q[discret_obs][action] += self.alpha*td_error



def train(agent, environment):
    best_reward = -float("inf")
    
    for episode in range(MAX_NUM_EPISODES):
        done = False
        truncate = False
        obs = environment.reset()[0]
        total_reward = 0.0
        
        while done == False and truncate == False: 
            action = agent.get_action(obs)
            
            next_obs, reward, done, truncate, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward
            
        if total_reward > best_reward:
            best_reward = total_reward
        
        print("Episodio número {} con recompensa: {}, mejor recompensa {}, epsilon: {}".format(episode, total_reward, best_reward, agent.epsilon))
    
    # devuelvo la mejor política de entrenamiento
    return np.argmax(agent.Q, axis = 2)



def test(agent, environment, policy):
    done = False
    truncate = False
    obs = environment.reset()[0]
    total_reward = 0.0
    
    while done == False and truncate == False: 
        environment.render()
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, truncate, info = environment.step(action)
        
        obs = next_obs
        total_reward += reward
        
    return total_reward


if __name__ == "__main__":
    environment = gym.make("MountainCar-v0", render_mode="rgb_array")
    environment.metadata["render_fps"]=50
    
    agent = QLearner(environment)
    
    learned_policy = train(agent, environment)
    monitor_path = "./monitor_output"
    environment = gym.wrappers.RecordVideo(environment, video_folder=monitor_path, 
                                           episode_trigger=lambda x: x % 100 == 0)
    environment.start_video_recorder()
    for _ in range(1000):
        
        test(agent, environment, learned_policy)
    environment.close_video_recorder()
    environment.close()


"""
environment = gym.make("MountainCar-v0")

agent = QLearner(environment)

learned_policy = train(agent, environment)

obs = environment.reset()[0]

discreteAction = agent.discretize(obs)

ohigh = agent.obs_high
olow = agent.obs_low
obin_width = agent.bin_width


tup_primero = obs-olow
tup_div = tup_primero/obin_width



step = 0
for _ in range(1000):
    discreteAction = agent.discretize(obs)
    action = agent.get_action(obs)
    next_obs, reward, done, truncate, info = environment.step(action)

    obs = next_obs
    
    ohigh = agent.obs_high
    olow = agent.obs_low
    obin_width = agent.bin_width
    
    
    tup_primero = obs-olow
    tup_div = tup_primero/obin_width
    
    tup_asyint = tup_div.astype(int)
    tupl = tuple(tup_asyint)
    step +=1
    #print(str(step )+ " - " + str(tupl))

"""














