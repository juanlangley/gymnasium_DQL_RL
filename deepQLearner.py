#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:12:50 2024

@author: juan
"""


import gymnasium as gym
import numpy as np
import os
import random
import gymnasium as gym


from datetime import datetime
from argparse import ArgumentParser

import torch

from brain import Brain
from brainCNN import BrainCNN

from decay_schedule import LinearDecaySchedule
from experience_memory import Experience, ExperienceMemory, ExperienceDeque
from params_manager import ParamsManager
from envs import atari
from envs import utils
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


args = ArgumentParser("deepQLearning")
args.add_argument("--params_file", help = "Path del fichero JSON de parámetros", 
                  default="parameters.json", metavar="PFILE")
args.add_argument("--env", help= "Entorno de ID de Atari en Gym", default = "SeaquestNoFreameskip-v4", metavar = "ENV")
args.add_argument("--gpu_id", help= "ID de la GPU a utilizar", default = 0,
                  type=int, metavar="GPU_ID")
args.add_argument("--test", help="Modo de testing", action="store_true", default=False)
args.add_argument("--render", help="Renderiza el entorno en pantalla", action="store_true", default=False)
args.add_argument("--record", help="Almacena videos y estados de la performance del agente", action="store_true", default=False)
args.add_argument("--output_dir", help= "Directorio para almacenar outputs", default = "./results")

args = args.parse_args()



manager = ParamsManager(args.params_file)
#ficheros de logs de configuración
summary_filename_prefix = manager.get_agent_params()["summary_filename_prefix"]
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

writer = SummaryWriter(summary_filename)

manager.export_agent_params(summary_filename + "/" + "agent_params.json")
manager.export_environment_params(summary_filename + "/" + "environment_params.json")





#contador de ejecuciones
global_step_num = 0

#habilitar gpu
use_cuda = manager.get_agent_params()["use_cuda"]
device = torch.device("cuda:"+ str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

#semilla para reproducir experimento
seed = manager.get_agent_params()["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)
    
    
    
class deepQLearner(object):
    def __init__(self, obs_shape, action_shape, params):
       
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['learning_rate']
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        self.training_steps_completed = 0
        self.action_shape = action_shape
        
        if len(obs_shape) == 1:
            # 1 dimensión de espacio de obs
            self.DQN = Brain
        elif len(obs_shape) == 3:
            # obs es imagen/3D
            self.DQN = BrainCNN

            
        self.Q = self.DQN(obs_shape, action_shape, device).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)
        
        if self.params['use_target_network']:
            self.Q_target = self.DQN(obs_shape, action_shape, device).to(device)
           
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params['epsilon_max']
        self.epsilon_min = self.params['epsilon_min']
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
                                                 max_steps = self.params['epsilon_decay_final_step'])
        self.step_num = 0
        
        self.memory = ExperienceMemory(capacity = int(self.params['experience_memory_size']))
        
         
    def get_action(self, obs):
        obs = np.array(obs)
        obs = obs / 255.0
        if len(obs.shape) == 3: # tenemos una imagen
            if obs.shape[2] < obs.shape[0]: # WxHxC -> C x H x W
                obs = obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
            obs = np.expand_dims(obs, 0)   
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
        self.step_num +=1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())   
        return action
        
        
    def learn(self, obs, action, reward, next_obs, done):
        if done:
            td_target = reward + 0.0
        else: 
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        
        self.Q_optimizer.step()
        
    def replay_experience(self, batch_size = None):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamaño de la muestra a tomar de la memoria
        :return: 
        """
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)   
        self.training_steps_completed += 1
      
    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fragmento de recuerdos anteriores
        :return: 
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)/255.0
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        
        if self.params["clip_reward"]:
            reward_batch = np.sign(reward_batch)
        next_obs_batch = np.array(batch_xp.next_obs)/255.0
        done_batch = np.array(batch_xp.done)

        if next_obs_batch.shape[3] < next_obs_batch.shape[1]:
            next_obs_batch = np.reshape(next_obs_batch, (next_obs_batch.shape[0], next_obs_batch.shape[1],next_obs_batch.shape[2]))
        if obs_batch.shape[3] < obs_batch.shape[1]:
            obs_batch = np.reshape(obs_batch, (obs_batch.shape[0], obs_batch.shape[1],obs_batch.shape[2]))

        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_frequency'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch *\
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        torch.max(self.Q_target(next_obs_batch),1)[0].data.tolist()
            td_target = torch.from_numpy(td_target)

        else: 
            td_target = reward_batch + ~done_batch * \
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        torch.max(self.Q(next_obs_batch).detach(),1)[0].data.tolist()
            td_target = torch.from_numpy(td_target)

        
        td_target = td_target.to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss(
                self.Q(obs_batch).gather(1, action_idx.view(-1,1)),
                td_target.float().unsqueeze(1))
        
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()
        
    def save(self, env_name):
        model_save_name = 'model.pt'
        #path = F"/content/drive/My Drive/{model_save_name}" 
        file_name = self.params['save_dir']+"DQL_"+env_name+".ptm"
        agent_state = {"Q": self.Q.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en : ", file_name)
        
        
    def load(self, env_name):
        #path = F"/content/drive/My Drive/trained_models/model.pt"
        file_name = self.params['load_dir']+"DQL_"+env_name+".ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Cargado del modelo Q desde", file_name,
              "que hasta el momento tiene una mejor recompensa media de: ",self.best_mean_reward,
              " y una recompensa máxima de: ", self.best_reward)



if __name__ == "__main__":
    env_conf = manager.get_environment_params()
    env_conf["env_name"] = args.env
    
    if args.test:
        env_conf["episodic_life"] = False
        env_conf["render_mode"] ="human"
    reward_type = "LIFE" if env_conf["episodic_life"] else "GAME"
        
    
    custom_region_available = False
    for key, value in env_conf["useful_region"].items():
        if key in args.env:
            env_conf["useful_region"] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_conf["useful_region"] = env_conf["useful_region"]["Default"]
    print("Configuración a utilizar:", env_conf)

    atari_env = False
    for game in atari.get_games_list():
        if game.replace("_", "") in args.env:
            tari_env = True
            
    if atari_env:
        environment = atari.make_env(args.env, env_conf)
    else:
        environment = utils.ResizeReshapeFrames(gym.make(args.env, render_mode= env_conf["render_mode"]))
    
    
    obs_shape = environment.observation_space.shape
    action_shape = environment.action_space.n
    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test
    agent_params["clip_reward"] = env_conf["clip_reward"]
    agent = deepQLearner(obs_shape, action_shape, agent_params)
    
    environment.metadata["render_fps"]=100
    

    
    episode_rewards = list()
    previous_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    
    if agent_params["load_trained_model"]:
        try:
            agent.load(env_conf["env_name"])
            previous_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("ERROR: No existe ningún modelo entrenado para este entorno. Empieza desde 0")
    
    episode = 0
    while global_step_num < agent_params["max_training_steps"]:
        obs, _ = environment.reset()
        total_reward = 0.0
        done = False
        step = 0
        
                
        while not done:
            if env_conf["render"] or args.render:
                environment.render()
            
            action = agent.get_action(obs)
            next_obs, reward, done, truncate, info = environment.step(action)
            
            """
            if done == True or truncate == True:
                done_resum = True
            else:
                done_resum = False
              """  
              
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            obs = next_obs
            total_reward += reward
            step += 1
            global_step_num += 1
            
            #if done is True or truncate is True:
            if done is True:
                episode += 1
                episode_rewards.append(total_reward)
            
                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward
                
                if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew: 
                    num_improved_episodes_before_checkpoint += 1
                
                if num_improved_episodes_before_checkpoint >= agent_params['save_freq']:
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0
                
                print("\n Episodio #{} finalizado con {} iteraciones. Con {} estados: recompensa = {}, recompensa media = {:.2f}, mejor recompensa = {}".
                      format(episode, step+1, reward_type, total_reward, np.mean(episode_rewards), agent.best_reward))
                
                writer.add_scalar("main/ep_reward", total_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_reward", agent.best_reward, global_step_num)
                
                if agent.memory.get_size() >= 2*agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()
                    
                break
            
    environment.close()
    writer.close()
    

    
"""
    
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