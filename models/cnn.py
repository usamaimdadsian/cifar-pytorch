
import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 13 * 13, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        outs = self.conv1(x)
        outs = self.relu(outs)
        outs = self.pool(outs)
    
        outs = torch.flatten(outs,1)
        outs = self.fc(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
