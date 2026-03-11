import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import math
import os
import pandas as pd

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            # if use_bn:
            #     layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # if use_bn:
                #     layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.GELU())  # Nonnegative activation
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """Initialize weights for linear layers"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass of DINO head

        Args:
            x: Input tensor of shape (batch_size, in_dim)

        Returns:
            x_proj: Projected features
            logits: Final logits after projection and normalization
        """
        x_proj = self.mlp(x)

        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits


class ContrastiveLearningViewGenerator(object):
    """Generate two random augmented views and one original view for contrastive learning"""

    def __init__(self, base_transform, ori_transform, n_views=2):
        self.base_transform = base_transform
        self.ori_transform = ori_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for _ in range(self.n_views-1)] + [self.ori_transform(x)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views-1)] + [self.ori_transform(x)]


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Also supports unsupervised SimCLR loss.
    From: https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for the model.
        If both `labels` and `mask` are None, it becomes SimCLR unsupervised loss.

        Args:
            features: Hidden vector with shape [bsz, n_views, ...]
            labels: Ground-truth labels [bsz]
            mask: Contrastive mask [bsz, bsz], mask_{i,j}=1 if same class

        Returns:
            Loss scalar
        """

        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` must be [bsz, n_views, ...], at least 3 dimensions')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):
    """Compute logits for InfoNCE loss used in contrastive learning"""

    b_ = int(features.size(0) / n_views)

    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # Remove diagonal
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives and negatives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    """Split parameters into regularized and non-regularized groups (no weight decay on bias/norm)"""
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    """Distillation loss used in DINO"""
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """Cross-entropy between soft teacher and student outputs"""
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # skip same view
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, labels):
        """Assume positive logit is always the first element"""
        loss = -x[:, 0] + torch.logsumexp(x[:, 1:], dim=1)
        return loss.mean()


class SimCLR(nn.Module):
    def __init__(self, temperature=0.5, n_views=2, contrastive=False):
        super(SimCLR, self).__init__()
        self.temp = temperature
        self.n_views = n_views

        if contrastive:
            self.criterion = contrastive_loss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, X):
        bs, n_dim = X.shape
        bs = int(bs / self.n_views)
        device = X.device

        labels = torch.cat([torch.arange(bs) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        similarity_matrix = torch.matmul(X, X.T)

        # Remove diagonal
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temp
        return logits, labels

    def forward(self, X):
        logits, labels = self.info_nce_loss(X)
        loss = self.criterion(logits, labels)
        return loss


class Z_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z_list = z.chunk(2, dim=0)
        z_sim = F.cosine_similarity(z_list[0], z_list[1], dim=1).mean()
        z_sim_out = z_sim.clone().detach()
        return -z_sim, z_sim_out


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative loss component"""
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, X):
        return self.compute_discrimn_loss(X.T)


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discrimn_loss(self, W):
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)

        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        # Supports both integer labels and probability membership vectors
        if len(Y.shape) == 1:
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for idx, label in enumerate(Y):
                Pi[label, 0, idx] = 1
        else:
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)

        total_loss = -discrimn_loss + self.gamma * compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]


class EntropyRegularizationLoss(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super(EntropyRegularizationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits):
        probabilities = F.softmax(logits, dim=1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        loss = torch.mean(self.alpha * entropy)
        return loss


def get_negative_mask(batch_size):
    """Mask for negative pairs in contrastive learning"""
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class DRO_Loss(nn.Module):
    """Distributionally Robust Optimization loss for contrastive learning"""
    def __init__(self, temperature, tau_plus, batch_size, beta, estimator, N=1.2e6, df=10):
        super(DRO_Loss, self).__init__()
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.batch_size = batch_size
        self.beta = beta
        self.estimator = estimator
        self.degrees_of_freedom = df

    def forward(self, out, index=None, labels=None):
        device = out.device

        if self.estimator == "adnce":
            out = F.normalize(out, dim=1)
            out_1, out_2 = torch.chunk(out, 2, dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            mask = get_negative_mask(self.batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)

            N = self.batch_size * 2 - 2
            mu = self.tau_plus
            sigma = self.beta
            weight = 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp(
                - (neg.log() * self.temperature - mu) ** 2 / (2 * sigma ** 2))
            weight = weight / weight.mean(dim=-1, keepdim=True)

            Ng = torch.sum(neg * weight.detach(), dim=1)
            loss = (-torch.log(pos / (pos + Ng))).mean()
            return loss, weight

        elif self.estimator == "weighted_nce_t":
            out = F.normalize(out, dim=1)
            out_1, out_2 = torch.chunk(out, 2, dim=0)
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
            mask = get_negative_mask(self.batch_size).to(device)
            neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)

            df = self.degrees_of_freedom
            mu = self.tau_plus
            sigma = self.beta

            t_values = (neg.log() * self.temperature - mu) / sigma
            t_dist_pdf = TDistributionPDF(df)
            weight = t_dist_pdf(t_values)
            weight = weight / weight.mean(dim=-1, keepdim=True)

            Ng = torch.sum(neg * weight.detach(), dim=1)
            loss = (-torch.log(pos / (pos + Ng))).mean()
            return loss, weight


class TDistributionPDF(nn.Module):
    def __init__(self, df):
        super(TDistributionPDF, self).__init__()
        self.df = df

    def forward(self, x):
        df = torch.tensor(self.df, dtype=torch.float32, device=x.device)
        gamma_term = torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2)
        coefficient = torch.exp(gamma_term) / torch.sqrt(df * math.pi)
        pdf = coefficient * (1 + x ** 2 / df) ** (-(df + 1) / 2)
        return pdf


class GroupSparseRegularization(torch.nn.Module):
    def __init__(self, a, lambda_):
        super(GroupSparseRegularization, self).__init__()
        self.a = a
        self.lambda_ = lambda_

    def forward(self, H):
        psi = lambda x: torch.log(self.a + x)
        penalty = torch.sum(torch.stack([psi(torch.norm(H[g], dim=1, p=1)) for g in range(len(H))]))
        return self.lambda_ * penalty


def sparseness(x):
    """Compute sparsity metric of feature vectors"""
    n = x.size(1)
    l1_norm = torch.norm(x, p=1, dim=1)
    l2_norm = torch.norm(x, p=2, dim=1)
    sparsity = ((math.sqrt(n) - l1_norm / l2_norm) / (math.sqrt(n) - 1)).mean()
    return 1 / sparsity


def orth(A, tol=1e-10):
    """Orthogonalize matrix using SVD"""
    U, S, V = torch.svd(A)
    num_effective_sv = (S > tol).sum().item()
    U = U[:, :num_effective_sv]
    return U


def custom_regularization(W, beta, gamma):
    """Custom regularization combining L1, L21, and Frobenius norm"""
    W_flatten = W.view(W.size(0), -1)
    l1 = torch.norm(W_flatten, 1)
    l21 = torch.sum(torch.sqrt(torch.sum(torch.square(W_flatten), dim=1)))
    lfro = torch.norm(W_flatten, 'fro')
    return beta * l1 + gamma * (l21 - lfro ** 2)