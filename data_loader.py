import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_dataloader(batch_size=128, validation_split=0.1, random_seed=42):
    """
    Prepares CIFAR-10 dataset for training, validation and testing.

    Args:
        batch_size (int): Number of samples per batch.
        validation_split (float): Proportion of the dataset to include in the validation split.
        random_seed (int): Seed for reproducibility of data splitting.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset

    print("Loading CIFAR-10 dataset...")
    train_val_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    print("CIFAR-10 dataset loaded successfully.")

    # Split training data into training and validation sets
    num_train = len(train_val_dataset)
    num_val = int(validation_split * num_train)
    num_train_actual = num_train - num_val

    # Ensure reproducibility
    generator = torch.Generator().manual_seed(random_seed)

    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [num_train_actual, num_val],
        generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Running CIFAR-10 data loader...")
    train_loader, val_loader, test_loader = get_cifar10_dataloader(batch_size=64)

    print("Checking: ")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break


