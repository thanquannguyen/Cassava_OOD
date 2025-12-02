import torch
import torch.nn as nn
from torchvision import models

class CassavaMobileNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(CassavaMobileNet, self).__init__()
        # Load MobileNetV3 Large
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Replace the classifier head
        # The original classifier is a Sequential block. 
        # We can check the in_features of the last linear layer.
        in_features = self.backbone.classifier[3].in_features
        
        # Keep the hardswish and dropout, just replace the last linear
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Test the model
    model = CassavaMobileNet(num_classes=5)
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
