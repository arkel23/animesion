import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight) 

class SEQLLoss(nn.Module):
    '''
    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).
    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.
    '''
    def __init__(self, freq_info, lamb=5e-3):
        super(SEQLLoss, self).__init__()
        self.freq_info = freq_info
        self.lamb = lamb
    
    def forward(self, pred_class_logits, gt_classes):
        self.pred_class_logits = pred_class_logits
        self.gt_classes = gt_classes
        return self.eql_loss()

    def eql_loss(self):

        self.n_i, self.n_c = self.pred_class_logits.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)

        eql_w = 1 - self.threshold_func() * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                        reduction='none')

        return torch.sum(cls_loss * eql_w) / self.n_i

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lamb] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

class SEQLLoss_beta(nn.Module):
    '''
    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).
    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.
    '''
    def __init__(self, freq_info, lamb=5e-3, gamma=0.95):
        super(SEQLLoss_beta, self).__init__()
        self.freq_info = freq_info
        self.lamb = lamb
        self.gamma = gamma
        #self.beta = np.random.binomial(1, gamma)
    
    def forward(self, pred_class_logits, gt_classes):
        self.pred_class_logits = pred_class_logits
        self.gt_classes = gt_classes
        return self.eql_loss()

    def eql_loss(self):

        self.n_i, self.n_c = self.pred_class_logits.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)

        beta = np.random.binomial(1, self.gamma)
        eql_w = 1 - beta * self.threshold_func() * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                        reduction='none')

        return torch.sum(cls_loss * eql_w) / self.n_i

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lamb] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight