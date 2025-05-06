import torch
#from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader



def mnistLoader(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Loading the Train and Test datasets
    trainDataSet = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    testDataSet = datasets.MNIST(root = './data', train = False, download = True, transform = transform)


    # Creating data loaders
    trainLoader = DataLoader(trainDataSet, batch_size = batch_size, shuffle = True)
    testLoader = DataLoader(testDataSet, batch_size = batch_size, shuffle = False)

    return trainLoader, testLoader

    """CIFAR-10 Data Loader
    """
def cifar10Loader(batch_size = 64):
    transform = transforms.Compose([
        transforms.toTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    trainDataSet = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    testDataSet = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)


    # Creating data loaders
    trainLoader = DataLoader(trainDataSet, batch_size = batch_size, shuffle = True)
    testLoader = DataLoader(testDataSet, batch_size = batch_size, shuffle = False)

    return trainLoader, testLoader

def main():
    choice = input("Select datasets (MNIST/CIFAR10): ").strip().lower()
    batch_input = input("Enter the batch size (default = 64): ")
    batchSize = int(batch_input) if batch_input else 64

    if choice == "mnist":
        trainLoader, testLoader = mnistLoader(batch_size = batchSize)
        print(f"MNIST: {len(trainLoader.dataset)} training samples, {len(testLoader.dataset)} test samples.")
    elif choice == "cifar10":
        trainLoader, testLoader = cifar10Loader(batch_size = batchSize)
        print(f"CIFAR-10: {len(trainLoader.dataset)} training samples, {len(testLoader.dataset)} test samples.")
    else:
        print("Invalid choice. Please choose between 'mnist' or 'cifar10'.")

if __name__ == '__main__':
    main()
        





