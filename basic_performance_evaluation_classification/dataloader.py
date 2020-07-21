import numpy as np
import tensorflow as tf

from random import shuffle
from tensorflow.keras.utils import to_categorical, Sequence

class GetDataset:
    def __init__(self, 
                 x_array, 
                 y_array, 
                 augment_fn=None,
                 do_balancing=False):
        """Get dataset and augment them.
        Args:
            - x_array (np.array): input images (N, H, W, C)
            - y_array (np.array): targets in one-hot format (N, c)
            - augment_fn (object): augment function
            - do_balancing (bool): do dataset balancing
        Return:
            x,y
        """
        self.augment_fn = augment_fn
        self.num_classes = y_array.shape[-1]
        if do_balancing:
            x_array, y_array = self._do_rebalance(x_array, y_array)

        # do shuffle
        tmp = list(zip(x_array, y_array))
        shuffle(tmp)
        x_array, y_array = zip(*tmp)
        self.x_array = x_array
        self.y_array = y_array

    def __getitem__(self, idx):
        idx = idx % self.__len__()

        img = self.x_array[idx]
        target = self.y_array[idx]

        if self.augment_fn:
            img = self.augment_fn.augment_image(img)
        img = np.float32(img) / 255.
        return img, target

    def __len__(self):
        return len(self.x_array)
    
    @staticmethod
    def _do_rebalance(x_array, y_array):
        num_per_class = y_array.sum(axis=0)
        y_flat = y_array.argmax(axis=-1)
        resample_index = [np.random.choice(np.where(y_flat==i)[0], 
                                           int(max(num_per_class))) for i in range(len(num_per_class))]
        resample_index = np.concatenate(resample_index)
        return x_array[resample_index], y_array[resample_index]

class DataLoader(Sequence):
    def __init__(self, dataset, batch_size):
    	"""Wrap images to batches
    	Args:
    		- dataset (object): dataset object from GetDataset
    		- batch_size (int): batch size
    	Return:
    		x, y in batches (N, H, W, C), (N, K)
    	"""
    	super(DataLoader, self).__init__()
    	self.dataset = dataset
    	self.batch_size = batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx+1) * self.batch_size
        imgs, targets = [], []
        for i in range(start_idx, end_idx):
            x_, y_ = self.dataset[i]
            imgs.append(x_)
            targets.append(y_)
        return np.array(imgs), np.array(targets)

    def __len__(self):
        return (len(self.dataset) // self.batch_size)