#!/usr/bin/env python
# coding: utf-8
from train import Train
t = Train(batch = 64,
          lr = 0.001,
          load_pretrain = False,
          model = 2,
          aug = True,
          mixup = True
         )
t.train(epoch_num = 100, step_one_epoch = 50, save_frq = 50, evl_frq = 50)
