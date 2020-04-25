import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.nn.functional as F



class AMCF(nn.Module):
    """
    The AMCF class
    """
    def __init__(self, num_user, num_item, num_asp=18, e_dim=16):
        super(AMCF, self).__init__()
        self.num_asp = num_asp # number of aspects
        self.user_emb = nn.Embedding(num_user, e_dim)
        self.item_emb = nn.Embedding(num_item, e_dim)

        self.u_bias = nn.Parameter(torch.randn(num_user))
        self.i_bias = nn.Parameter(torch.randn(num_item))
        self.asp_emb = Aspect_emb(num_asp, e_dim)
        self.mlp = nn.Sequential(nn.Linear(e_dim, 50), nn.Linear(50, 25), nn.Linear(25, num_asp))
        self.e_dim = e_dim
        self.pdist = nn.PairwiseDistance(p=2) # used to calculate pairwise distance

    def forward(self, x, y, asp):
        user_latent = self.user_emb(x)
        item_latent = self.item_emb(y)
        detached_item_latent = item_latent.detach() # gradient shielding

        u_bias = self.u_bias[x]
        i_bias = self.i_bias[y]
        
        out = (user_latent * item_latent).sum(-1) + u_bias + i_bias + 3.53 #4.09, 3.53
        asp_latent = self.asp_emb(asp) # [batch_size, num_asp, e_dim]
        # asp_weight = torch.bmm(asp_latent, item_latent.unsqueeze(-1)) # [batch, num_asp, 1]
        # asp_weight = F.softmax(asp_weight, dim=1)
        asp_weight = F.sigmoid(self.mlp(detached_item_latent)).unsqueeze(-1)
        item_asp = torch.bmm(asp_latent.permute(0,2,1), asp_weight).squeeze(-1)
        
        # cosine similarity between item_latent and item_asp
        # sim = - F.cosine_similarity(item_asp, detached_item_latent, dim=-1)

        # Euclidian distance
        sim = self.pdist(item_asp, detached_item_latent)

        # get preference output [num_asp]
        user_latent = user_latent.unsqueeze(-1)
        pref = torch.bmm(asp_latent, user_latent).squeeze(-1) #[batch, num_asp]

        return [out, sim, pref]

    def predict_pref(self, x):
    
        user_latent = self.user_emb(x)
        user_latent = user_latent.unsqueeze(1).expand(-1, self.num_asp, -1)
        # virtual item latent
        batch_size = x.shape[0]
        dummy_asp = torch.ones([batch_size, self.num_asp]).cuda()
        item_latent = self.asp_emb(dummy_asp) # [batch, num_asp, e_dim]
        user_latent = user_latent.reshape(-1, self.e_dim)
        item_latent = item_latent.reshape(-1, self.e_dim)
        
        out = (user_latent * item_latent).sum(-1)# + 3.53
        out = out.reshape(-1, self.num_asp)
        out = F.normalize(out, p=1, dim=-1)
        return out
    
    
    def predict_spec_pref(self, x, y, asp):
    
        user_latent = self.user_emb(x)
        user_latent = user_latent.unsqueeze(1).expand(-1, self.num_asp, -1)
        item_latent = self.item_emb(y)
        item_latent = item_latent.unsqueeze(1).expand(-1, self.num_asp, -1)

        user_latent = user_latent.reshape(-1, self.e_dim)
        item_latent = item_latent.reshape(-1, self.e_dim)
        
        out = (user_latent * item_latent).sum(-1)# + 3.53
        out = out.reshape(-1, self.num_asp)
        out = F.normalize(out, p=1, dim=-1)
        return out

class Aspect_emb(nn.Module):
    """
    module to embed each aspect to the latent space.
    """
    def __init__(self, num_asp, e_dim):
        super(Aspect_emb, self).__init__()
        self.num_asp = num_asp
        self.W = nn.Parameter(torch.randn(num_asp, e_dim)) # contains all aspect latents

    def forward(self, x):
        # note that x  are multi-hot vectors of dimension [batch_size, num_asp]
        shape = x.shape
        x = x.reshape([x.shape[0], x.shape[1], 1])
        x = x.expand(-1, -1, self.W.shape[1])
        asp_latent = torch.mul(x, self.W) # [batch_size, num_asp, e_dim]
        # we must normalize asp_latent per aspect
        asp_latent = F.normalize(asp_latent, p=2, dim=2)
        

        return asp_latent