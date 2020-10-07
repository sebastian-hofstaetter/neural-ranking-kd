import torch.nn as nn
import torch

class MarginMSELoss(nn.Module):
    def __init__(self):
        super(MarginMSELoss, self).__init__()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
        All inputs should be tensors of equal size
        """
        loss = torch.mean(torch.pow((scores_pos - scores_neg) - (label_pos - label_neg),2))
        return loss