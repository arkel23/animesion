import numpy as np
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities as utilities

batch_size = 16
no_epochs = 50
steps_per_epoch = 1000
total_steps = no_epochs * steps_per_epoch

cdwu_percent = 0.2
cdwu_steps = int(total_steps*cdwu_percent)

total_seq_len = 80
max_text_seq_len = 16

mask_scheduler = utilities.scheduler.MasksSchedule('sigmoid',
    batch_size, total_seq_len, max_text_seq_len, cdwu_steps, cdwu_steps, total_steps)

nonzero_percent_list = []

for step in range(total_steps):

    mask = mask_scheduler.ret_mask([step])
    mask_text = mask[:, -max_text_seq_len:]

    mask_text_len = mask_text.shape[1] * mask_text.shape[0]
    nonzero_mask_text_len = torch.count_nonzero(mask_text).item()

    nonzero_percent = ((mask_text_len - nonzero_mask_text_len) / mask_text_len) * 100
    nonzero_percent_list.append(nonzero_percent)
    #print(nonzero_text_mask_len, mask_text_len, mask_nonzero_percent)

#print(len(nonzero_percent_list))
plt.plot(np.arange(total_steps), nonzero_percent_list)
plt.ylim([0, 101])
plt.xlabel('Global step')
plt.ylabel('Tokens (text) masked (%)')
plt.title('Percentage of tokens masked as function of training progress')
plt.show()
