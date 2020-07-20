from gym.spaces import Space, Discrete, Box

import numpy as np


def get_space_size(space: Space):
    space_type = type(space)
    if space_type == Discrete:
        return space.n
    elif space_type == Box:
        return np.prod(space.shape)
