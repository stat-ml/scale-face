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


class MLS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu_X, log_sigma, cos_func=False, **kwargs):
        mu_X = F.normalize(mu_X)
        sigma_sq_X = torch.exp(log_sigma)
        if cos_func:
            func = (
                lambda f1, f2: -torch.einsum("ij,dj->idj", f1, f2)
                / (f1.norm(dim=-1)[:, None] @ f2.norm(dim=-1)[None] + 1e-5)[..., None]
            )
        else:
            func = lambda f1, f2: (f1.unsqueeze(1) - f2.unsqueeze(0)) ** 2

        sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)

        diff = func(mu_X, mu_X) / (1e-10 + sig_sum) + torch.log(sig_sum)
        diff = diff.sum(dim=2, keepdim=False)
        return -diff


class MLS_lfw(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cos_func=False, **kwargs):
        mu_X, log_sigma = kwargs["feature"], kwargs["log_sigma"]
        mu_X = F.normalize(mu_X)
        sigma_sq_X = torch.exp(log_sigma)
        if cos_func:
            func = (
                lambda f1, f2: -torch.einsum("ij,dj->idj", f1, f2)
                / (f1.norm(dim=-1)[:, None] @ f2.norm(dim=-1)[None] + 1e-5)[..., None]
            )
        else:
            func = lambda f1, f2: (f1.unsqueeze(1) - f2.unsqueeze(0)) ** 2

        sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)

        diff = func(mu_X, mu_X) / (1e-10 + sig_sum) + torch.log(sig_sum)
        diff = diff.sum(dim=2, keepdim=False)
        return -diff

class MLSLoss(FaceModule):
    """
    TODO: docs
    """

    def __init__(self, mean=False, **kwargs):
        super(MLSLoss, self).__init__(kwargs)
        self.mean = mean

    def forward(self, device, mu_X, gty, log_sigma):
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int().to(gty.device)
        loss_mat = -MLS()(mu_X, log_sigma)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        pos_loss = loss_mat[pos_mask].mean()
        return pos_loss

class ProbConstraintLoss(FaceModule):
    def __init__(self):
        super(ProbConstraintLoss, self).__init__()

    def forward(self, **kwargs):
        log_sigma = kwargs["log_sigma"]
        log_sigma_mean = log_sigma.mean(0)
        loss = torch.norm(log_sigma / log_sigma_mean - 1.0, p=1, dim=-1).mean()
        return loss


class ProbTripletLoss(FaceModule):
    def __init__(self):
        super(ProbTripletLoss, self).__init__()

    def forward(self, **kwargs):
        mu_X, log_sigma = kwargs["feature"], kwargs["log_sigma"]
        sigma_sq_X = torch.exp(log_sigma)
        sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
        func = lambda f1, f2: (f1.unsqueeze(1) - f2.unsqueeze(0)) ** 2
        diff = func(mu_X, mu_X) / (1e-10 + sig_sum) + torch.log(sig_sum)
        return diff


class ProbLoss(FaceModule):
    """
    TODO: docs
    """

    def __init__(self, m=3, lambda_c=0.1, lambda_id=0.1, **kwargs):
        super(ProbLoss, self).__init__(kwargs)
        self.m = m
        self.lambda_c = lambda_c
        self.lambda_id = lambda_id

    def forward(self, device, **kwargs):
        mu_X, gty, log_sigma = kwargs["feature"], kwargs["gty"], kwargs["log_sigma"]
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int().to(gty.device)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        nty_mask = (1 - gty_mask).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        neg_mask = (non_diag_mask * nty_mask) > 0

        loss_mls = -MLS()(mu_X, log_sigma)
        loss_mls = loss_mls[pos_mask].mean()

        loss_c = self.lambda_c * ProbConstraintLoss()(**kwargs)

        loss_triplet = ProbTripletLoss()(**kwargs)

        triplet_loss = torch.cat(
            (loss_triplet[pos_mask], -loss_triplet[neg_mask]), dim=0
        ) + self.m

        triplet_loss = (
            self.lambda_id
            * torch.max(
                torch.zeros(1, device=triplet_loss.device), triplet_loss
            ).mean()
        )

        return loss_mls + loss_c + triplet_loss
