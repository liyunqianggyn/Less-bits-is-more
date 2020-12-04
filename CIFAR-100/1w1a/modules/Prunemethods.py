import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.autograd as autograd
import numpy as np
from args import args as parser_args


"""

Different Prune strategies = ['Sign', 'Bihalf', 'Sign_OurPrune', 'Bihalf_OurPrune','Bihalf_OurPrune_rescale']

"""


class Signonly(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        
        out =  scores.sign()                     

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    

    
class Bihalfonly(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        
        # Get the signmask by sorting the scores magnitude and using the top k% as sign(score) +1, -1, remaining as 0
        out = scores.clone()
        a =scores[0].nelement()
        j = int((1 -0.5) * scores[0].nelement())   

        # bi-half
        flat_out = out.view(out.size(0), -1)                                           # convert to matrix with c_out * (c_in *k*k)
        sort_fea,index_fea = scores.view(scores.size(0), -1).sort()                    # sort over each channel
        B_creat = np.concatenate((-torch.ones([scores.size(0), j]), \
                                  torch.ones([scores.size(0), a-j])), 1) 
        B_creat = torch.FloatTensor(B_creat).cuda()
        out =(torch.zeros(flat_out.size()).cuda()).scatter_(1, index_fea, B_creat)     # assign the values to construct signmask 
        out = out.view(scores.size())                     
        
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None    


class SignOurPruneGetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        
        # Get the subnetwork by sorting the weight scores and prune k 
        out = scores.abs().clone()
        _, idx = scores.abs().flatten().sort()
        j = int(k * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        out = out * scores.sign()                     
        
        return out


    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class BihalfOurPruneGetSubnet(autograd.Function): 
    @staticmethod
    def forward(ctx, scores, k):
        # Get the signmask by sorting the weight scores and using the top 1-k/2 as 1, bottom 1-k/2 as -1, remaining as 0
        out = scores.clone()
        _, idx = scores.view(scores.size(0), -1).sort()
        a =scores[0].nelement()
        
        j = int((1 - (1-k)/2.) * scores[0].nelement())
        i = int((1-k)/2. * scores[0].nelement())       

        # Bi-half
        flat_out = out.view(out.size(0), -1)                                 # convert to matrix with c_out * (c_in *k*k)
        sort_fea,index_fea = scores.view(scores.size(0), -1).sort()          # sort over each channel
        B_creat = np.concatenate((-torch.ones([scores.size(0), i]), \
                                  torch.zeros([scores.size(0), j - i]),\
                                  torch.ones([scores.size(0), a-j])), 1) 
        B_creat = torch.FloatTensor(B_creat).cuda()
        out =(torch.zeros(flat_out.size()).cuda()).scatter_(1, index_fea, B_creat)  # assign the values to construct signmask 
        out = out.view(scores.size())                        

        
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None 


class HPConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(HPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # Define a scale factor per layer, then expand as weights size 
        self.scale = nn.Parameter(torch.Tensor(self.weight.size()))  
        self.scale.requires_grad = False       
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        gain = nn.init.calculate_gain(parser_args.nonlinearity)
        std = gain / math.sqrt(fan)
        self.scale.data = torch.ones_like(self.weight.data) * std        
 
       
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        
    def forward(self, x):
        
        if parser_args.masksigntype == 'Sign':   
            subnet = Signonly.apply(self.weight, parser_args.prune_rate)
            w = self.scale * subnet     
            
        elif parser_args.masksigntype == 'Bihalf':          
            subnet = Bihalfonly.apply(self.weight, parser_args.prune_rate)
            w = self.scale * subnet     
            
        elif parser_args.masksigntype == 'Sign_OurPrune':  
            subnet = SignOurPruneGetSubnet.apply(self.weight, parser_args.prune_rate)
            w = self.scale * subnet    
           
        elif parser_args.masksigntype == 'Bihalf_OurPrune':          
            subnet = BihalfOurPruneGetSubnet.apply(self.weight, parser_args.prune_rate)           
            w = self.scale * subnet   
           
        elif parser_args.masksigntype == 'Bihalf_OurPrune_rescale':          
            subnet = BihalfOurPruneGetSubnet.apply(self.weight, parser_args.prune_rate)
            # define a rescale factor
            rescale =  1 / math.sqrt(1 - parser_args.prune_rate)                       
            w = self.scale * rescale * subnet   


        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x        
