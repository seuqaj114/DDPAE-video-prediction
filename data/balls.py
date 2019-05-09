import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data

def load_fixed_set(path, is_train):
  dataset = np.load(path)
  if is_train:
    dataset = dataset["train_x"]
  else:
    dataset = dataset["test_x"]
  return dataset

class Balls3bp(data.Dataset):

  def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
               transform=None):
    '''
    param num_objects: a list of number of possible objects.
    '''
    super(Balls3bp, self).__init__()

    if is_train:
      path = "/home/miguel/Documents/prog/phd/bunnyhop/data/datasets/balls/color_3bp_vx2_vy2_sl20_r2_g60_m1_dt05.npz"
    else:
      path = "/home/miguel/Documents/prog/phd/bunnyhop/data/datasets/balls/color_3bp_vx2_vy2_sl40_r2_g60_m1_dt05.npz"

    self.dataset = load_fixed_set(path, is_train)
    self.length = int(1e4) if self.dataset is None else self.dataset.shape[0]

    self.is_train = is_train
    self.num_objects = num_objects
    self.n_frames_input = n_frames_input
    self.n_frames_output = n_frames_output
    self.n_frames_total = self.n_frames_input + self.n_frames_output
    self.transform = transform

  def __getitem__(self, idx):
    length = self.n_frames_input + self.n_frames_output
    images = self.dataset[idx, ...]

    if self.transform is not None:
      images = self.transform(images)
    input = images[:self.n_frames_input]
    if self.n_frames_output > 0:
      output = images[self.n_frames_input:length]
    else:
      output = []

    return input, output

  def __len__(self):
    return self.length


class BallsSpring(data.Dataset):

  def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
               transform=None):
    '''
    param num_objects: a list of number of possible objects.
    '''
    super(BallsSpring, self).__init__()

    if is_train:
      path = "/home/miguel/Documents/prog/phd/bunnyhop/data/datasets/balls/color_spring_vx8_vy8_sl12_r2_k4_e6.npz"
    else:
      path = "/home/miguel/Documents/prog/phd/bunnyhop/data/datasets/balls/color_spring_vx8_vy8_sl12_r2_k4_e6.npz"
    self.dataset = load_fixed_set(path, is_train)
    self.length = int(1e4) if self.dataset is None else self.dataset.shape[0]

    self.is_train = is_train
    self.num_objects = num_objects
    self.n_frames_input = n_frames_input
    self.n_frames_output = n_frames_output
    self.n_frames_total = self.n_frames_input + self.n_frames_output
    self.transform = transform

  def __getitem__(self, idx):
    length = self.n_frames_input + self.n_frames_output
    images = self.dataset[idx, ...]

    if self.transform is not None:
      images = self.transform(images)
    input = images[:self.n_frames_input]
    if self.n_frames_output > 0:
      output = images[self.n_frames_input:length]
    else:
      output = []

    return input, output

  def __len__(self):
    return self.length