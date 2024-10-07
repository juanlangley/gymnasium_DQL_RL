#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 08:36:20 2024

@author: Juan
"""

import gymnasium as gym
import sys


MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
environment = gym.make("ALE/Assault-v5", render_mode="human")
environment.metadata["render_fps"]=500

for episode in range(MAX_NUM_EPISODES):
    obs = environment.reset()

    for step in range(MAX_STEPS_PER_EPISODE):
        environment.render()

        action = environment.action_space.sample()
        
        next_state, reward, done, truncate, info = environment.step(action)
        #print(str(next_state)+ "\n " + str(reward) + "\n " + str(truncate) + "\n " + str(done) + "\n " + str(info))
        obs = next_state
        
        if done is True or truncate is True: 
            print("\n Episodio #{} terminado en {} steps".format(episode, step+1))
            break
environment.close()


"""
action = agent.choose_action(obs)
next_state, reward, done, info = environment.step(action)
obs = next_state
"""



"""
def run_gym_environment(argv):
    # Nombre del juego/entorno
    environment = gym.make(argv[1], render_mode="human")
    environment.reset()
    for _ in range(int(argv[2])):
        environment.render()
        environment.step(environment.action_space.sample())
        
    environment.close()


if __name__ == "__main__":
    run_gym_environment(sys.argv)
"""

"""
environment = gym.make("ALE/SpaceInvaders-ram-v5", render_mode="human")
environment.reset()
for _ in range(1000):
    environment.render()
    environment.step(environment.action_space.sample())
    
environment.close()
"""  