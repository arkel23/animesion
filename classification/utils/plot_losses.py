import os
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(training_proc_avg, test_proc_avg, results_dir):
    # to plot learning curves
    x = np.arange(1, len(training_proc_avg)+1)
    x_2 = np.linspace(1, len(training_proc_avg)+1, len(test_proc_avg))

    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x, training_proc_avg, label='Training loss')
    axs.plot(x_2, test_proc_avg, label='Testing loss')
    axs.set_xlabel('Epoch no.')
    axs.set_ylabel('Average loss for epoch')
    axs.set_title('Loss as training progresses')
    axs.legend()
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fig.savefig(os.path.join(results_dir, 'training_loss.png'), dpi=300)

