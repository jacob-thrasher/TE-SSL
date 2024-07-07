import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.autograd import Variable
from lib.utils import pad_col

class CoxLoss(nn.Module):
    '''
    This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    '''
    def __init__(self):
        super(CoxLoss, self).__init__()
    
    def forward(self, hazard_pred, survtime, censor):
        device = (hazard_pred.get_device())

        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = survtime[j] >= survtime[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss_cox


class DeepHitSingleLoss(nn.Module):
    '''
    From: https://github.com/havakv/pycox/blob/master/pycox/models/loss.py

    Loss for DeepHit (single risk) model [1].
    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    
    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)

    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum': sum.

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    '''
    def __init__(self, alpha=0.5, sigma=0.1, reduction='mean', device='cuda'):
        super(DeepHitSingleLoss, self).__init__()

        self.alpha = alpha
        self.sigma = sigma
        self.reduction = reduction
        self.device = device

    def _pair_rank_mat(self, mat, idx_durations, events, dtype='float32'):
        n = len(idx_durations)
        for i in range(n):
            dur_i = idx_durations[i]
            ev_i = events[i]
            if ev_i == 0:
                continue
            for j in range(n):
                dur_j = idx_durations[j]
                ev_j = events[j]
                if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                    mat[i, j] = 1
        return mat

    def pair_rank_mat(self, idx_durations, events, dtype='float32'):
        """        
        Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
        So it takes value 1 if we observe that i has an event before j and zero otherwise.
        
        Arguments:
            idx_durations {np.array} -- Array with durations.
            events {np.array} -- Array with event indicators.
        
        Keyword Arguments:
            dtype {str} -- dtype of array (default: {'float32'})
        
        Returns:
            np.array -- n x n matrix indicating if i has an observerd event before j.
        """
        idx_durations = idx_durations.reshape(-1)
        events = events.reshape(-1)
        n = len(idx_durations)
        mat = np.zeros((n, n), dtype=dtype)
        mat = self._pair_rank_mat(mat, idx_durations, events, dtype)
        return mat  

    def _reduction(self, loss: Tensor, reduction: str = 'mean') -> Tensor:
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

    def _diff_cdf_at_time_i(self, pmf: Tensor, y: Tensor) -> Tensor:
        """
        R is the matrix from the DeepHit code giving the difference in CDF between individual
        i and j, at the event time of j. 
        I.e: R_ij = F_i(T_i) - F_j(T_i)
        
        Arguments:
            pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
            y {torch.tensor} -- Matrix with indicator of duration/censor time.
        
        Returns:
            torch.tensor -- R_ij = F_i(T_i) - F_j(T_i)
        """
        n = pmf.shape[0]
        ones = torch.ones((n, 1), device=pmf.device)
        r = pmf.cumsum(1).matmul(y.transpose(0, 1))
        diag_r = r.diag().view(1, -1)
        r = ones.matmul(diag_r) - r
        return r.transpose(0, 1)

    def nll_pmf(self, phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean',
            epsilon: float = 1e-7) -> Tensor:
        """
        Negative log-likelihood for the PMF parametrized model [1].
        
        Arguments:
            phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
            idx_durations {torch.tensor} -- Event times represented as indices.
            events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
                Same length as 'idx_durations'.
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum: sum.
        
        Returns:
            torch.tensor -- The negative log-likelihood.

        References:
        [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
            with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
            https://arxiv.org/pdf/1910.06724.pdf
        """
        if phi.shape[1] <= idx_durations.max():
            raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                            f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                            f" but got `phi.shape[1] = {phi.shape[1]}`")
        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1)
        idx_durations = idx_durations.view(-1, 1)
        phi = pad_col(phi)
        gamma = phi.max(1)[0]
        cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
        sum_ = cumsum[:, -1]
        part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
        part2 = - sum_.relu().add(epsilon).log()
        part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
        # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
        loss = - part1.add(part2).add(part3)
        return self._reduction(loss, reduction)

    def _rank_loss_deephit(self, pmf: Tensor, y: Tensor, rank_mat: Tensor, sigma: float,
                        reduction: str = 'mean') -> Tensor:
        """Ranking loss from DeepHit.
        
        Arguments:
            pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
            y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
            rank_mat {torch.tensor} -- See pair_rank_mat function.
            sigma {float} -- Sigma from DeepHit paper, chosen by you.
        
        Returns:
            torch.tensor -- loss
        """
        r = self._diff_cdf_at_time_i(pmf, y)
        loss = rank_mat * torch.exp(-r/sigma)
        loss = loss.mean(1, keepdim=True)
        return self._reduction(loss, reduction)

    def rank_loss_deephit_single(self, phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor,
                             sigma: Tensor, reduction: str = 'mean') -> Tensor:
        """Rank loss proposed by DeepHit authors [1] for a single risks.

        Arguments:
            phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
                all in (-inf, inf).
            idx_durations {torch.tensor} -- Int tensor with index of durations.
            events {torch.tensor} -- Float indicator of event or censoring (1 is event).
            rank_mat {torch.tensor} -- See pair_rank_mat function.
            sigma {float} -- Sigma from DeepHit paper, chosen by you.
        
        Keyword Arguments:
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum': sum.
        
        Returns:
            torch.tensor -- Rank loss.

        References:
        [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
            approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
            Intelligence, 2018.
            http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
        """
        idx_durations = idx_durations.view(-1, 1)
        # events = events.float().view(-1)
        pmf = pad_col(phi).softmax(1)
        y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.) # one-hot
        rank_loss = self._rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
        return rank_loss

    # TODO: Figure out if rank_mat is a required arg
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
            '''
            Arguments:
                phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
                    all in (-inf, inf).
                idx_durations {torch.tensor} -- Int tensor with index of durations.
                events {torch.tensor} -- Float indicator of event or censoring (1 is event).
                rank_mat {torch.tensor} -- See pair_rank_mat function.
            '''
            idx_durations = [t / 6 for t in idx_durations] # Convert to indicies
            idx_durations = torch.tensor(idx_durations).type(torch.int64).to(self.device)
            rank_mat = torch.tensor(self.pair_rank_mat(idx_durations, events)).to(self.device)

            nll = self.nll_pmf(phi, idx_durations, events, self.reduction)
            rank_loss = self.rank_loss_deephit_single(phi, idx_durations, events, rank_mat, self.sigma,
                                                self.reduction)
            return self.alpha * nll + (1. - self.alpha) * rank_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
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
                raise ValueError('Num of labels does not match num of features')
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class RegularizedCoxLoss(nn.Module):
    '''
    Adds (supervised) contrastive loss as regularization term

    Args:
        beta (float) - Weight for regularization term
        temperature (float) - temperature value for constrastive loss
        contrast_mode (str)
        base_temperature (float) - 
        do_supcon (bool) - If TRUE, include event indiactors as supervised contrastive labels
                           If FALSE, perform unsupervised contrastive loss calculation
    '''
    def __init__(self, alpha=1, beta=0.5, temperature=0.07, contrast_mode='all', do_supcon=False):
        super(RegularizedCoxLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.coxloss = CoxLoss()
        self.reg = SupConLoss(temperature=temperature, contrast_mode=contrast_mode)
        self.do_supcon = do_supcon
    
    def forward(self, pred, t, e, features, mask=None):
        '''
        Args:
            pred: hazard prediction of shape [bsz]
            t: event/censor time
            e: event indicator (and label for supcon algorithm)
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        '''

        if type(pred) is tuple:
            cox = self.coxloss(pred[0], t, e) + self.coxloss(pred[1], t, e)
        elif type(pred) is torch.Tensor:
            cox = self.coxloss(t, e, pred) 
        else: 
            raise TypeError(f'Expected pred to be of type torch.Tensor or Tuple of tensors, got {type(pred)}')


        if not self.do_supcon: e = None

        reg = self.reg(features, labels=e, mask=mask)
        return self.alpha*cox + self.beta*reg

# This could be directly implemented into SupConLoss, but I wanted to preserve it just in case
class TESSL_Loss(nn.Module):
    """
    TE-SSL Loss
    
    Based on Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, alpha=1, beta=0.5, compare_time_with_self=False):
        super(TESSL_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.alpha = alpha
        self.beta = beta
        self.compare_time_with_self = compare_time_with_self

    def forward(self, features, labels=None, mask=None, times=None):
        """
        Compute loss for model. 
        
        If `times` is None, Computes SupCon loss
        If `times`, `labels`, and `mask` are None, Computes SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif times is not None and labels is None:
            raise ValueError('Cannot define "times" without defining "labels"')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if times is not None:
            # Compute deltas
            a = times.tile((batch_size, 1))
            b = times.contiguous().view(-1, 1).tile(1, batch_size)
            deltas = torch.abs(a - b)

            # Get min and max of all positive pairs
            # inverted mask lets us alter only negative pairs
            inverted_mask = ((~torch.eq(labels, labels.T)).float())
            if self.compare_time_with_self:
                _min = torch.min(deltas + inverted_mask*1e10)
                _max = torch.min(deltas - inverted_mask*1e10)
            else:
                eye = torch.eye(batch_size).to(device)
                _mask = (inverted_mask + eye)*1e10
                _min = torch.min(deltas*mask + _mask)
                _max = torch.max(deltas*mask - _mask)

            m = (self.alpha - self.beta) / (_min - _max)
            b = (((self.beta - self.alpha) / (_min - _max))*_min) + self.alpha
            all_weights = (m * deltas) + b      # Weights for all pairs
            mask = all_weights * mask  # weights for only positive pairs, torch.equal(mask, (weighted_mask != 0).float()) should be true

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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        log_prob_mask = logits_mask

        if times is not None:
            unmasked_weights = all_weights.repeat(anchor_count, contrast_count)
            log_prob_mask = unmasked_weights * logits_mask 

        # compute log_prob
        exp_logits = torch.exp(logits) * log_prob_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = (mask != 0).float().sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
