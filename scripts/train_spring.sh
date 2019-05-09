#!/bin/bash
python train.py \
  --gpus 0 \
  --n_workers 4 \
  --ckpt_dir $HOME/Documents/prog/phd/ddpae/ckpt \
  --dset_dir $HOME/Documents/prog/phd/ddpae \
  --dset_name spring \
  --evaluate_every 1 \
  --lr_init 1e-3 \
  --lr_decay 1 \
  --n_iters 200000 \
  --batch_size 64 \
  --n_components 2 \
  --stn_scale_prior 2 \
  --ckpt_name spring \
  --n_frames_input 3 \
  --n_frames_output 7 \
  --save_every 10