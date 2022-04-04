import numpy as np
import random
import copy
import datetime
import platform
from numpy.core.fromnumeric import mean
import torch
import torch.nn.functional as F
from torch.serialization import load
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel






a = torch.Tensor([0,0,0,0,1])