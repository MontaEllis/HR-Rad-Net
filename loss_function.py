from torch import nn
import torch
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Classification_Loss(nn.Module):
    def __init__(self):
        super(Classification_Loss, self).__init__()
        self.focal_criterion = FocalLoss()
        self.mse_criterion = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()


    def forward(self, model_output, targets, model):

        regularization_loss = 0
        for param in model.module.parameters():
            regularization_loss += torch.sum(torch.abs(param)) #+torch.sum(torch.abs(param))**2
        loss = self.focal_criterion(F.sigmoid(model_output), targets) #+ 0.00001 * regularization_loss

        loss += self.bce(model_output,targets)
        return loss
