import math
import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class MasksSchedule():

    def __init__(self, mask_schedule, batch_size, total_seq_len, max_text_seq_len, 
        warmup_steps, cooldown_steps, total_steps, cycles=.5):
        self.mask_schedule = mask_schedule
        
        self.batch_size = batch_size
        self.total_seq_len = total_seq_len
        self.max_text_seq_len = max_text_seq_len
        self.image_seq_len = self.total_seq_len - self.max_text_seq_len
        
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self.cycles = cycles
    
    def ret_mask(self, step):
        step = step[0]
        if self.mask_schedule == None:
            return None
        
        elif self.mask_schedule == 'constant':
            # 15 % masking like bert but only mask (0) or 1
            mask_images = np.ones((self.batch_size, self.image_seq_len))
            mask_text = np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), p=[0.15, 0.85])
            return torch.from_numpy(np.hstack((mask_images, mask_text)))
        
        elif self.mask_schedule == 'sigmoid':
            # during warmup attend to all tokens
            # during cooldown attend to no text tokens
            # else attend to a percentage of text tokens following cosine function
            mask_images = np.ones((self.batch_size, self.image_seq_len))
            
            if step < self.warmup_steps:
                mask_text = np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), p=[0, 1])
            
            elif step > (self.total_steps - self.cooldown_steps):
                mask_text = np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), p=[1, 0])
            
            else:
                progress = (float(step - self.warmup_steps) / 
                    (float(max(1, self.total_steps - self.warmup_steps - self.cooldown_steps))))
                
                prob_visible = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
                prob_mask = 1.0 - prob_visible
                
                mask_text = np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), 
                    p=[prob_mask, prob_visible])

            return torch.from_numpy(np.hstack((mask_images, mask_text)))   
