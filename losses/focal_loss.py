
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights = None
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.Tensor(per_cls_weights)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        per_cls_weights = None

        if self.weight is not None:
            per_cls_weights = torch.FloatTensor(self.weight).cuda()
            

        # calculate softmax probabilities
        input_soft = F.softmax(input, dim=1)

        # get the predicted probability for the ground-truth class
        batch_size, num_classes = input.size()
        class_mask = input.data.new(batch_size, num_classes).fill_(0)
        class_mask.scatter_(1, target.view(-1, 1), 1.)
        if input.is_cuda and not per_cls_weights.is_cuda:
            per_cls_weights = per_cls_weights.cuda()
        if per_cls_weights is not None:
            per_cls_weights = per_cls_weights.view(1, -1)
            class_mask = class_mask * per_cls_weights

        probs = (input_soft * class_mask).sum(1).view(-1, 1)

        # calculate focal loss
        log_p = probs.log()
        loss = -1 * (1 - probs) ** self.gamma * log_p
        loss = torch.mean(loss,dim=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
