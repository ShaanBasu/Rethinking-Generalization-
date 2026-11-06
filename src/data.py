import torchvision
import numpy as np
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