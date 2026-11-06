import torchvision
def load_cifar10(filePath):
    train_dataset = torchvision.datasets.CIFAR10(
        root = filePath, train = True, download = True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root = filePath, train = False, download = True
    )
    
    return train_dataset, test_dataset
    