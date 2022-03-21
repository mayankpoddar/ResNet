import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, in_planes, num_layers, num_blocks, num_classes=10):
        if not isinstance(num_blocks, list):
            raise Exception("num_blocks parameter should be a list of integer values")
        if num_layers != len(num_blocks):
            raise Exception("Residual layers should be equal to the length of num_blocks list")
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.num_layers = num_layers
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        for i in range(2, num_layers+1):
            setattr(self, "layer"+str(i), self._make_layer(block, 2*self.in_planes, num_blocks[i-1], stride=2))
        finalshape = list(getattr(self, "layer"+str(num_layers))[-1].modules())[-2].num_features
        self.linear = nn.Linear(finalshape, num_classes)
        self.path = "./project1_model.pt"

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        custom_layers = []
        for stride in strides:
            custom_layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*custom_layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(1, self.num_layers+1):
            out = eval("self.layer" + str(i) + "(out)")
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def saveToDisk(self):
        torch.save(self.state_dict(), self.path)

    def loadFromDisk(self):
        self.load_state_dict(torch.load(self.path))

if __name__ == "__main__":
    model = ResNet(BasicBlock, 42, 4, [2, 2, 2, 2])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_parameters)
    x = torch.rand(1, 3, 32, 32)
    model(x)
