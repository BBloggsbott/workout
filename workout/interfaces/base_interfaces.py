import gym
import numpy as np
import torch.nn as nn

from workout.utils.env_utils import get_space_size

class StateVectorEnvironemtInterface(object):
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.__env = gym.make(env_name)
        self.__action_space_size = get_space_size(self.__env.action_space)
        self.__observation_space_size = get_space_size(self.__env.observation_space)
        self.__model = nn.Linear(self.__observation_space_size, self.__action_space_size)

        self.__train_fn = lambda env, model, input_preprocessor, output_preprocessor, episodes:None

    def __input_preprocessor(self, model_input):
        return model_input             # returns processed data

    def __output_preprocessor(self, model_output):
        return model_output


    def train(self, episodes: int):
        self.__train_fn(self.__env, self.__model, self.__input_preprocessor, self.__output_preprocessor, episodes)

    def get_train_fn(self):
        return self.__train_fn

    def set_train_fn(self, train_fn):
        self.__train_fn = train_fn

    def get_env(self):
        return self.__env

    def set_env(self, env):
        self.__env = env

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model
        

