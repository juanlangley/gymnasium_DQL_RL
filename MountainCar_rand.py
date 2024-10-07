#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:42:54 2024

@author: Juan
"""

import gymnasium as gym
import sys


environment = gym.make("MountainCar-v0", render_mode="human")
environment.metadata["render_fps"]=500


MAX_NUM_EPISODES = 1000



for episode in range(MAX_NUM_EPISODES):
    done = False
    truncate = False
    obs = environment.reset()
    total_reward = 0.0
    step = 0
    
    while done == False and truncate == False:
        environment.render()
        
        # Get Action
        action = environment.action_space.sample()
        
        next_state, reward, done, truncate, info = environment.step(action)
        
        total_reward += reward
        print(obs)
        obs = next_state
        step += 1
    
    print("\n Episodio número {} finalizado con {} iteraciones. Recompensa final= {}".format(episode, step+1, total_reward))

environment.close()

        