#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:57:38 2024

@author: Juan
"""


import gymnasium as gym
import numpy as np
import os
import random
import gymnasium as gym

import torch
from brain import Brain
from decay_schedule import LinearDecaySchedule
from experience_memory import Experience, ExperienceMemory, ExperienceDeque

MAX_NUM_EPISODES = 100000
STEPS_PER_EPISODE = 300


class SwallowQLearner(object):
    def __init__(self, environment, learning_rate = 0.005, gamma = 0.98):
        self.obs_shape = environment.observation_space.shape[0]
        self.action_shape = environment.action_space.n
        self.gamma = gamma
        
        self.Q = Brain(self.obs_shape, self.action_shape)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min,
                                                 max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        self.memory = ExperienceDeque(capacity=int(1e5))

    def get_action(self, obs):
        return self.policy(obs)
    
    
    def epsilon_greedy_Q(self, obs):
        
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.torch(torch.device(self.device)).numpy())
        return action
        
        
    def learn(self, obs, action, reward, next_obs):
        


        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)

        
        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()
        
    
    def replay_experience(self, batch_size):
        #experience_batch = self.memory.sample(batch_size)
        experience_batch = self.memory.random_batch(batch_size)
        self.learn_from_batch_experience(experience_batch)
        
        
    def learn_from_batch_experience(self, experiences):
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).detach().max(1)[0].data.numpy()
                
        
        td_target = torch.from_numpy(td_target).to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        
        td_error = torch.nn.functional.mse_loss(
            self.Q(obs_batch).gather(1, action_idx.view(-1,1)),
            td_target.float().unsqueeze(1))

        self.optimizer.zero_grad()
        td_error.mean().backward()
        self.optimizer.step()




if __name__ == "__main__":
    environment = gym.make("CartPole-v1", render_mode="rgb_array")
    environment.metadata["render_fps"]=50
    
    agent = SwallowQLearner(environment)
    
    first_episode = True
    done_resum = False
    episode_rewards = list()
    max_reward = 0.0
    
    for episode in range(MAX_NUM_EPISODES):
        obs, _ = environment.reset()
        total_reward = 0.0

        
        for step in range(STEPS_PER_EPISODE):
            action = agent.get_action(obs)
            
            next_obs, reward, done, truncate, info = environment.step(action)
            
            if done == True or truncate == True:
                done_resum = True
            else:
                done_resum = False
            agent.memory.store(Experience(obs, action, reward, next_obs, done_resum))
            agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward
            
            
            if done is True or truncate is True:
                if first_episode:
                    max_reward = total_reward
                    first_episode = False
                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("\n Episodio {} finaliado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                
                if agent.memory.get_size() > 1000:
                    agent.replay_experience(320)
                
                break
    environment.close()





"""
environment = gym.make("CartPole-v0", render_mode="rgb_array")
environment.metadata["render_fps"]=50

obs_shape = environment.observation_space.shape[0]

agent = SwallowQLearner(environment)


"""












