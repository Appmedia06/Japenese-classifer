import torch.nn as nn

class Classifer(nn.Module):
  def __init__(self, input_ch=1, output_ch=50):
    super(Classifer, self).__init__()

    self.cnn1 = nn.Sequential(
        nn.Conv2d(input_ch, 32, kernel_size=5, stride=1, padding=1), # (1, 84, 84) -> (32, 84, 84)
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0), # (32, 84, 84) -> (32, 42, 42)
    )

    self.cnn2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1), # (32, 42, 42) -> (64, 42, 42)
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0), # (64, 42, 42) -> (64, 21, 21)
    )

    self.cnn3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1), # (64, 21, 21) -> (128, 21, 21)
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0), # (128, 21, 21) -> (128, 10, 10)
    )

    self.fc = nn.Sequential(
        nn.Linear(in_features=128 * 8 * 8, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=output_ch)
    )

  def forward(self, x):
    # x = self.pretrained_model(x)
    x = self.cnn1(x)
    x = self.cnn2(x)
    x = self.cnn3(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
  

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if x.size() != out.size():
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


# 定义ResNet-9模型
class ResNet9(nn.Module):
    def __init__(self, num_classes=71):
        super(ResNet9, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = self.make_block(16, 16, stride=1)
        self.block2 = self.make_block(16, 32, stride=2)
        self.block3 = self.make_block(32, 64, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def make_block(self, in_channels, out_channels, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out



class JA_NN(nn.Module):
  def __init__(self, num_classes=71):
    super(JA_NN, self).__init__()
    
    self.conv_layer1 = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Dropout(0.3)
    )

    self.conv_layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Dropout(0.3)    
    )

    self.flatten = nn.Flatten()

    self.fc = nn.Sequential(
      nn.Linear(64 * 4 * 4, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, num_classes)
    )

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.conv_layer1(x)

    x = self.conv_layer2(x)
    
    x = self.flatten(x)

    x = self.fc(x)

    # x = self.softmax(x)
    return x