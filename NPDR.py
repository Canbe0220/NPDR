import copy
import torch
import math
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.distributions import Categorical
from mlp import MLPCritic, MLPActor

class OCE(nn.Module):
    def __init__(self, in_dim, out_dim, num_mem_slots=16, proj_drop=0.):
        super(OCE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_mem_slots = num_mem_slots
        
        self.h1 = nn.Linear(in_dim, out_dim, bias=False)
        
        self.mk = nn.Linear(out_dim, num_mem_slots, bias=False)
        self.mv = nn.Linear(num_mem_slots, out_dim, bias=False)

        self.h2 = nn.Linear(out_dim, out_dim, bias=False)

        self.proj_drop = nn.Dropout(proj_drop)

        if in_dim != out_dim:
            self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_fc = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.h1.weight)
        nn.init.xavier_normal_(self.mk.weight)
        nn.init.xavier_normal_(self.mv.weight)
        nn.init.xavier_normal_(self.h2.weight)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight)

    def forward(self, x):
        B, N, C = x.shape
        h_x = self.h1(x)

        attn = self.mk(h_x)
        attn = F.softmax(attn, dim=-2)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.mv(attn)

        h_x = self.h2(attn)
        h_x = self.proj_drop(h_x)

        h_res = self.res_fc(x)
        
        return h_x + h_res


class MCE(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0.):
        super(MCE, self).__init__()
        self.ope_dim = in_dim[0]
        self.mac_dim = in_dim[1]
        self.out_dim = out_dim
        self.nega_slope = 0.2

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.ope_w = nn.Linear(self.ope_dim, self.out_dim, bias=False)
        self.mac_w = nn.Linear(self.mac_dim, self.out_dim, bias=False)
        
        self.ope_alpha = nn.Parameter(torch.empty(size=(self.out_dim, 1)))
        self.mac_alpha = nn.Parameter(torch.empty(size=(self.out_dim, 1)))    
        
        self.leaky_relu = nn.LeakyReLU(self.nega_slope)

        if self.mac_dim != out_dim:
            self.res_fc = nn.Linear(self.mac_dim, out_dim, bias=False)
        else:
            self.res_fc = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.nega_slope)
        nn.init.xavier_normal_(self.ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.mac_w.weight, gain=gain)
        nn.init.xavier_normal_(self.ope_alpha, gain=gain)
        nn.init.xavier_normal_(self.mac_alpha, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight)
        
    def forward(self, curr_proc_batch, batch_idxes, feats):
        
        feat_ope = self.feat_drop(feats[0])
        feat_mac = self.feat_drop(feats[1])
            
        h_ope = self.ope_w(feat_ope) 
        h_mac = self.mac_w(feat_mac)
        
        attn_ope = torch.matmul(h_ope, self.ope_alpha).squeeze(-1)
        attn_mac = torch.matmul(h_mac, self.mac_alpha).squeeze(-1)
        
        attn_ope = attn_ope.unsqueeze(-1) + attn_mac.unsqueeze(-2) 
        e_ijk = self.leaky_relu(attn_ope)

        mask_ijk = (curr_proc_batch[batch_idxes] == 1)
        e_ijk = e_ijk.masked_fill(~mask_ijk, float('-9e10'))
        
        alpha_ijk = F.softmax(e_ijk, dim=-2)
        alpha_ijk = torch.where(mask_ijk, alpha_ijk, torch.zeros_like(alpha_ijk))
        alpha_ijk = self.attn_drop(alpha_ijk)
        
        out_mac = torch.matmul(alpha_ijk.transpose(1, 2), h_ope)

        if self.res_fc is not None:
             out_res = self.res_fc(feat_mac)
        else:
             out_res = feat_mac

        return out_mac + out_res


class NPDR(nn.Module):
    def __init__(self, model_paras):
        super(NPDR, self).__init__()

        self.device = model_paras["device"]
        self.in_ope_dim = model_paras["in_ope_dim"] 
        self.in_mac_dim = model_paras["in_mac_dim"]  
        self.in_pair_dim = model_paras["in_pair_dim"]
        self.out_ope_dim = model_paras["out_ope_dim"]
        self.out_mac_dim = model_paras["out_mac_dim"]
        self.out_pair_dim = model_paras["out_pair_dim"]
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        self.actor_in_dim = model_paras["actor_in_dim"]
        self.critic_in_dim = model_paras["critic_in_dim"]
        self.actor_layer = self.critic_layer = model_paras["policy_layer"]
        self.actor_hidden_dim = self.critic_hidden_dim = model_paras["policy_hidden_dim"] 
        self.actor_out_dim = self.critic_out_dim = model_paras["policy_out_dim"] 
        
              
        self.get_opes = OCE(self.in_ope_dim, self.out_ope_dim)
        self.get_macs = MCE((self.out_ope_dim, self.in_mac_dim), self.out_mac_dim, self.dropout, self.dropout)


        self.actor = MLPActor(self.actor_layer, self.actor_in_dim, self.actor_hidden_dim, self.actor_out_dim).to(self.device)
        self.critic = MLPCritic(self.critic_layer, self.critic_in_dim, self.critic_hidden_dim, self.critic_out_dim).to(self.device)


    def act_prob(self, state, memory, flag_train=True, flag_sample=True):

        '''
        probability distribution
        '''

        curr_proc_adj = state.curr_proc_batch
        batch_idxes = state.batch_idxes
        raw_opes = state.feat_opes_batch[batch_idxes]
        raw_macs = state.feat_macs_batch[batch_idxes]
        
        # Normalize
        min_opes = torch.min(raw_opes, dim=-2, keepdim=True)[0]
        max_opes = torch.max(raw_opes, dim=-2, keepdim=True)[0]
        norm_opes = (raw_opes - min_opes) / (max_opes - min_opes + 1e-8)

        min_macs = torch.min(raw_macs, dim=-2, keepdim=True)[0]
        max_macs = torch.max(raw_macs, dim=-2, keepdim=True)[0]
        norm_macs = (raw_macs - min_macs) / (max_macs - min_macs + 1e-8)

        h_opes = self.get_opes(norm_opes)
        h_macs = self.get_macs(curr_proc_adj[..., 0], batch_idxes, (h_opes, norm_macs)) 
        h_pair = curr_proc_adj[batch_idxes]
     
        h_opes_pooled = h_opes.mean(dim=-2)
        h_macs_pooled = h_macs.mean(dim=-2)

        # expand and concatenate
        h_opes_expand = h_opes.unsqueeze(-2).expand(-1, -1, h_macs.size(-2), -1)
        h_macs_expand = h_macs.unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_expand = h_opes_pooled[:, None, None, :].expand_as(h_opes_expand)
        h_macs_pooled_expand = h_macs_pooled[:, None, None, :].expand_as(h_macs_expand)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)
        candidate_opes = ~(state.mask_job_procing_batch[batch_idxes] + state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_opes_expand[..., 0])
        idle_macs = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_opes_expand[..., 0])
        mask_proc = (curr_proc_adj[batch_idxes, ..., 0] == 1) & candidate_opes & idle_macs

        # actor MLP
        # h_actions = torch.cat((h_opes_expand, h_macs_expand, h_opes_pooled_expand, h_macs_pooled_expand, h_pair), dim=-1).transpose(1, 2)
        h_actions = torch.cat((h_opes_expand, h_macs_expand, h_pair), dim=-1).transpose(1, 2)
        mask = mask_proc.transpose(1, 2).flatten(1)
        
        #priority probability
        prob = self.actor(h_actions).flatten(1)
        prob[~mask] = float('-inf')
        action_probs = F.softmax(prob, dim=1)

        h_pooled = torch.cat((h_opes_pooled, h_macs_pooled), dim=-1)
        values = self.critic(h_pooled)    

        if flag_sample:
            # using sample strategy during training
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        else:
            # using greedy strategy during validating and testing
            action_indexes = action_probs.argmax(dim=1)
        
        if flag_train == True:
            # Store memory data during training
            memory.logprobs.append(dist.log_prob(action_indexes))
            memory.action_indexes.append(action_indexes)
            memory.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memory.curr_proc_adj.append(copy.deepcopy(curr_proc_adj))
            memory.norm_opes.append(copy.deepcopy(norm_opes))
            memory.norm_macs.append(copy.deepcopy(norm_macs))
            memory.mask_proc.append(copy.deepcopy(mask_proc))
            memory.values.append(copy.deepcopy(values.squeeze()))
            
        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / curr_proc_adj.size(1)).long()
        jobs = (action_indexes % curr_proc_adj.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]         

        return torch.stack((opes, mas, jobs), dim=1).t()


    def evaluate(self, curr_proc_adj, norm_opes, norm_macs, mask_proc, action_indexes):
        batch_idxes = torch.arange(0, curr_proc_adj.size(0)).long()
        features = (norm_opes, norm_macs)

        h_opes = self.get_opes(norm_opes)
        h_macs = self.get_macs(curr_proc_adj[..., 0], batch_idxes, (h_opes, norm_macs)) 
        h_pair = curr_proc_adj[batch_idxes]

        h_opes_pooled = h_opes.mean(dim=-2)
        h_macs_pooled = h_macs.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_opes_expand = h_opes.unsqueeze(-2).expand(-1, -1, h_macs.size(-2), -1)
        h_macs_expand = h_macs.unsqueeze(-3).expand(-1, h_opes.size(-2), -1, -1)
        h_opes_pooled_expand = h_opes_pooled[:, None, None, :].expand_as(h_opes_expand)
        h_macs_pooled_expand = h_macs_pooled[:, None, None, :].expand_as(h_macs_expand)

        # h_actions = torch.cat((h_opes_expand, h_macs_expand, h_opes_pooled_expand, h_macs_pooled_expand, h_pair), dim=-1).transpose(1, 2)
        h_actions = torch.cat((h_opes_expand, h_macs_expand, h_pair), dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_macs_pooled), dim=-1)
        prob = self.actor(h_actions).flatten(1)
        mask = mask_proc.transpose(1, 2).flatten(1)

        prob[~mask] = float('-inf')
        action_probs = F.softmax(prob, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_indexes)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys