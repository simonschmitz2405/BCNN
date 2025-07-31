import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

from blitz.modules import BayesianConv2d, BayesianLinear
from torchvision.models.resnet import BasicBlock

class BaselineResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineResNet18, self).__init__()
        # Load a pre-trained ResNet-18 model 
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adjustment for CIFAR-10 dataset ###

        # Adjustment for first convolution layer (conv1) for smaller 32*32 input images
        # The original ResNet-18's conv1 is designed for 224x224 ImageNet images
        # and has kernel_size=7, stride=2, padding=3.
        # For CIFAR-10 (32x32), a smaller kernel and stride often work better,
        # or even removing the initial max-pooling layer (which we'll do here to keep it simple
        # as the original ResNet for CIFAR-10 paper often skips it).
        # We replace the first conv layer to be more suitable for CIFAR-10 (32x32 input images)
        # Typically, for CIFAR, ResNet implementations use kernel_size=3, stride=1, padding=1 for conv1
        # and often skip the maxpool layer immediately after.
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride =1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # Remove maxpool layer

        # Adjust the final fully connected layer to match CIFAR-10's 10 classes
        # The original ResNet-18's fc layer outputs 1000 classes (for ImageNet).
        # We need to replace it with a new linear layer that outputs `num_classes`.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class BayesianResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(BayesianResNet18, self).__init__()

        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modification for CIFAR-10 dataset
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool layer

        # Bayesian layers (Replace the final fully connected layer and the last convolutional layer)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = BayesianLinear(num_ftrs, num_classes)

        self.resnet.layer4[1].conv2 = BayesianConv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=1,
            padding=1,
        )

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":

    # # Instantiate the model
    # model = BaselineResNet18(num_classes=10)
    # print(model)

    # dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    # print("Dummy input shape:", dummy_input.shape)

    # # Pass dummy input through the model
    # output = model(dummy_input)
    # print(f"Output shape: {output.shape}")  # Should be [1, 10] for CIFAR-10

    # # Verify the number of parameters
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params:,}")  

    # # Test with a larger batch size
    # dummy_batch = torch.randn(64, 3, 32, 32)
    # output_batch = model(dummy_batch)
    # print(f"\nDummy batch input shape: {dummy_batch.shape}")
    # print(f"Output batch shape: {output_batch.shape}") # Expected: torch.Size([64, 10])

    # print("\nBaselineResNet18 model implementation complete and successfully tested for demonstration.")


    # Bayesian ResNet-18 model
    bayesian_model = BayesianResNet18(num_classes=10)
    print(bayesian_model)

    print("\nModified layer check:")
    print(f"Final FC layer: {bayesian_model.resnet.fc}")
    print(f"Last Conv layer: {bayesian_model.resnet.layer4[1].conv2}")

    # Test with a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)
    print(f"\nDummy input shape: {dummy_input.shape}")
    
    # Perform multiple forward passes to demonstrate uncertainty
    print("Performing 5 forward passes to demonstrate varying outputs:")
    for i in range(5):
        output = bayesian_model(dummy_input)
        print(f"Pass {i+1}: Output logits = {output.detach().numpy()}")
        
    total_params = sum(p.numel() for p in bayesian_model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    print("\nBayesianResNet18 model implementation complete and successfully tested.")



