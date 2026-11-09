import torchvision
import numpy as np
import torch
def load_cifar10(filePath):
    train_dataset = torchvision.datasets.CIFAR10(
        root = filePath, train = True, download = True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root = filePath, train = False, download = True
    )
    
    return train_dataset, test_dataset

def subset_10000_maker(train_set, seed, per_class = 1000):
    label_array = np.array(train_set.targets)
    rng = np.random.default_rng(seed)
    chunks= []
    for c in range(10):
        indices_with_label_c = np.where(label_array == c)[0]
        randomly_chosen_indices = rng.choice(indices_with_label_c, size = per_class, replace = False)
        chunks.append(randomly_chosen_indices)
    
    return np.sort(np.concatenate(chunks))

def prepare_data(train_set, test_set, subset_ind):
    
    #1. We select the subset from the training_set and take everything from test_set
    train_images = train_set.data[subset_ind] # Shape: (10000, 32, 32, 3) uint8
    train_labels = np.array(train_set.targets)[subset_ind] # Shape: (10000,)
    test_images = test_set.data # Shape: (10000, 32, 32, 3)
    test_labels = np.array(test_set.targets) # Shape: (10000,)
    
    # 2. We then convert the numpy array shape of the train and test images into float tensors with channels first 
    x_train = torch.from_numpy(train_images).float().div(255).permute(0, 3, 1, 2) # permute basically rearrganes the axes, 0- number of images, 1-height, 2- width, 3 - colour channels
    x_test = torch.from_numpy(test_images).float().div(255).permute(0, 3, 1, 2)
    
    
    # 3. We then get the per-channel statistics from only the Training Subset
    mean = x_train.mean(dim = (0, 2, 3)) # the dim functions basically collapses all axes except for the colour channel ones, Shape: (3,)
    std = x_train.std(dim = (0, 2, 3)) # Shape: (3,)
    
    # 4. We then normalise both sets with the mean and std
    mean_b = mean.view(1, 3, 1, 1) 
    std_b = std.view(1, 3, 1, 1)
    x_train = ((x_train - mean_b)/std_b).contiguous() # the contiguous() rearranges the tensor in memory to match its shape
    x_test = ((x_test - mean_b)/std_b).contiguous()
    
    # 5. We then convert the labels as int64 tensors
    y_train = torch.from_numpy(train_labels).long() #.long() just converts it into a 64-bit integer
    y_test = torch.from_numpy(test_labels).long()
    
    return x_train, y_train, x_test, y_test, mean, std