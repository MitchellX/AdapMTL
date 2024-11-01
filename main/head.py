import torch
import torch.nn as nn

from models.Deeplab_ResNet34 import STRConv, initialize_sInit

# Heads for Pixel2Pixel
# Note: Should create for each task
#       Should be able to connect with the backbone model
class ASPPHeadNode(nn.Module):
    def __init__(self, feature_channels, out_channels):
        super(ASPPHeadNode, self).__init__()
        
        # sparseThreshold for every task
        self.sparseThreshold = nn.Parameter(initialize_sInit())

        self.fc1 = Classification_Module(feature_channels, out_channels, rate=6, sparseThreshold=self.sparseThreshold)
        self.fc2 = Classification_Module(feature_channels, out_channels, rate=12, sparseThreshold=self.sparseThreshold)
        self.fc3 = Classification_Module(feature_channels, out_channels, rate=18, sparseThreshold=self.sparseThreshold)
        self.fc4 = Classification_Module(feature_channels, out_channels, rate=24, sparseThreshold=self.sparseThreshold)

    def forward(self, x):
        output = self.fc1(x) + self.fc2(x) + self.fc3(x) + self.fc4(x)  # concate 4 FC results as output
        return output
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        return
    
class Classification_Module(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12, sparseThreshold=None):
        super(Classification_Module, self).__init__()
        self.conv1 = STRConv(inplanes, 1024, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True, sparseThreshold=sparseThreshold)
        self.conv2 = STRConv(1024, 1024, kernel_size=1, sparseThreshold=sparseThreshold)
        self.conv3 = STRConv(1024, num_classes, kernel_size=1, sparseThreshold=sparseThreshold)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        return