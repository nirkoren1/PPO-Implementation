import torch
import torch.nn as nn
import torch.nn.functional as F

class No_clip_Loss(nn.Module):
    def __init__(self, **kwargs):
        super(No_clip_Loss, self).__init__()

    def forward(self, ratio, advantages, **kwargs):
        return -(ratio * advantages).mean()

class Clip_Loss(nn.Module):
    def __init__(self, clip_coef, **kwargs):
        super(Clip_Loss, self).__init__()
        self.clip_coef = clip_coef

    def forward(self, ratio, advantages, **kwargs):
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        return -torch.min(pg_loss1, pg_loss2).mean()
    
class KL_Penalty_Loss(nn.Module):
    def __init__(self, beta, **kwargs):
        super(KL_Penalty_Loss, self).__init__()
        self.beta = beta

    def forward(self, ratio, advantages, kl, **kwargs):
        return -(advantages * ratio - self.beta * kl).mean()
    
class Value_Loss(nn.Module):
    def __init__(self, vf_coef, **kwargs):
        super(Value_Loss, self).__init__()
        self.vf_coef = vf_coef
        
    def forward(self, new_values, returns, **kwargs):
        return self.vf_coef * F.mse_loss(new_values, returns)