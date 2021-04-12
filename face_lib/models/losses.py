import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from face_lib.models import FaceModule


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(FaceModule):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, **kwargs):
        cosine, gty = kwargs["cosine"], kwargs["label"]
        index = torch.where(gty != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, gty[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine


class AngleLoss(FaceModule):
    """
    https://arxiv.org/pdf/1704.08063.pdf
    """

    def __init__(self, gamma=0, **kwargs):
        super(AngleLoss, self).__init__(kwargs)
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, **kwargs):
        target = kwargs.get("gty")
        input = kwargs.get("angle_x")

        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))

        output = cos_theta * 1.0

        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


class MLSLoss(FaceModule):
    """
    TODO: docs
    """

    def __init__(self, mean=False, **kwargs):

        super(MLSLoss, self).__init__(kwargs)
        self.mean = mean

    def negMLS(self, mu_X, sigma_sq_X):
        if self.mean:
            XX = torch.mul(mu_X, mu_X).sum(dim=1, keepdim=True)
            YY = torch.mul(mu_X.T, mu_X.T).sum(dim=0, keepdim=True)
            XY = torch.mm(mu_X, mu_X.T)
            mu_diff = XX + YY - 2 * XY
            sig_sum = sigma_sq_X.mean(dim=1, keepdim=True) + sigma_sq_X.T.sum(
                dim=0, keepdim=True
            )
            diff = mu_diff / (1e-8 + sig_sum) + mu_X.size(1) * torch.log(sig_sum)
            return diff
        else:
            mu_diff = mu_X.unsqueeze(1) - mu_X.unsqueeze(0)
            sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
            diff = torch.mul(mu_diff, mu_diff) / (1e-10 + sig_sum) + torch.log(
                sig_sum
            )  # BUG
            diff = diff.sum(dim=2, keepdim=False)
            return diff

    def forward(self, **kwargs):
        mu_X, gty, log_sigma = kwargs["feature"], kwargs["gty"], kwargs["log_sigma"]
        mu_X = F.normalize(mu_X)  # if mu_X was not normalized by l2
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if gty.device.type == "cuda":
            non_diag_mask = non_diag_mask.cuda(0)
        sig_X = torch.exp(log_sigma)
        loss_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        pos_loss = loss_mat[pos_mask].mean()
        return pos_loss
