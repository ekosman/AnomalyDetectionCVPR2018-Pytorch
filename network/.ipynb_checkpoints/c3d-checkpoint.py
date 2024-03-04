""""This module contains an implementation of C3D model for video
processing."""

import itertools

import torch
from torch import Tensor, nn


class C3D(nn.Module):
    """The C3D network."""

    def __init__(self, pretrained=None):
        super().__init__()

        self.pretrained = pretrained

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)
        )

        self.fc6 = nn.Linear(8192, 4096)
        self.relu = nn.ReLU()
        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x: Tensor):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))

        return x

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = [
            # Conv1
            "conv1.weight",
            "conv1.bias",
            # Conv2
            "conv2.weight",
            "conv2.bias",
            # Conv3a
            "conv3a.weight",
            "conv3a.bias",
            # Conv3b
            "conv3b.weight",
            "conv3b.bias",
            # Conv4a
            "conv4a.weight",
            "conv4a.bias",
            # Conv4b
            "conv4b.weight",
            "conv4b.bias",
            # Conv5a
            "conv5a.weight",
            "conv5a.bias",
            # Conv5b
            "conv5b.weight",
            "conv5b.bias",
            # fc6
            "fc6.weight",
            "fc6.bias",
        ]

        ignored_weights = [
            f"{layer}.{type_}"
            for layer, type_ in itertools.product(["fc7", "fc8"], ["bias", "weight"])
        ]

        p_dict = torch.load(self.pretrained)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                if name in ignored_weights:
                    continue
                print("no corresponding::", name)
                continue
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        """Initialize weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    inputs = torch.ones((1, 3, 16, 112, 112))
    net = C3D(pretrained=False)

    outputs = net.forward(inputs)
    print(outputs.size())
