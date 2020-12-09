import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_stats(samples_per_class, no_classes, class_names):
	samples_per_class = samples_per_class.groupby('class_id', as_index=True).count()['dir'].squeeze()
		
	set_mean = samples_per_class.mean()
	set_median = samples_per_class.median()
	set_std = samples_per_class.std()
	print('Dataset no of classes: {}\n'
	'Mean number of samples per class: {}\n'
	'Median number of samples per class: {}\n'
	'Standard deviation of samples per class: {}'.format(
	no_classes, set_mean, set_median, set_std))

	samples_per_class_ordered = samples_per_class.sort_values(0, ascending=False)
	print(samples_per_class_ordered.head())
	print('Shape of samples per class df:')
	print(samples_per_class_ordered.shape)

	print('Characters with most number of samples: ')
	print('\t'.join('No. {}: {} (Class ID: {}) with {} samples'.format(
    j+1, class_names[samples_per_class_ordered.index[j], 0], 
	class_names[samples_per_class_ordered.index[j], 1], 
	samples_per_class_ordered.iloc[j]) for j in range(10)))

	print('\nCharacters with least number of samples: ')
	print('\t'.join('No. {}: {} (Class ID: {}) with {} samples'.format(
    j+1, class_names[samples_per_class_ordered.index[-1-j], 0], 
	class_names[samples_per_class_ordered.index[-1-j], 1], 
	samples_per_class_ordered.iloc[-1-j]) for j in range(10)))

	fig, axs = plt.subplots(1)
	fig.suptitle('Histogram of Classes for DanbooruFace Dataset')

	# only plot first 100
	bins = 100
	axs.bar(np.arange(bins), samples_per_class_ordered.iloc[0:bins].to_numpy())
		
	axs.set_ylabel('No. of samples per class')
	axs.set_title('Ordered based on no. of samples')
		
	results_dir = 'results'
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	fig.savefig(os.path.join(results_dir, 
	'histogram_danbooruFacesCrops.png'), dpi=300)
		