#!/bin/bash
python test.py \
  --gpus 0 \
  --n_workers 4 \
  --batch_size 16 \
  --dset_name moving_mnist \
  --ckpt_dir $HOME/Documents/prog/phd/ddpae/ckpt \
  --dset_dir $HOME/Documents/prog/phd/ddpae \
  --log_every 5 \
  --save_visuals 0 \
  --save_all_results 1 \
  --ckpt_name crop_NC2_lr1.0e-03_bt64_200k \
  --n_frames_input 4 \
  --n_frames_output 15
