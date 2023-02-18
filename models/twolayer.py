
import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = torch.reshape(x,(x.shape[0],self.input_dim))
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
