import torch


def get_mask(state):
    """
    Generates a mask for nonexistent components
    identified by the state being a constant vector of -1
    """
    padding_mask = torch.all(state.eq(-1), 2).unsqueeze(-1)
    return padding_mask

EPS = 1e-16

class AdsorptiveAttnBlock(torch.nn.Module):
    '''
    Scaled Dot-Product Attention
    Make K and V learnable parameters
    '''

    def __init__(self, n_comp, scale, n_site=1):
        # dim_q is the size of query vector
        super().__init__()
        self.scale = scale
        self.n_site = n_site
        self.n_comp = n_comp
        self.softmax = torch.nn.Softmax(dim=3)
    
        # learnable attention value: (n_comp * n_comp)
        self.layer_v = torch.nn.Conv1d(n_site * n_comp, 
            n_site * n_comp, n_comp, groups=n_site * n_comp,
            bias=False)

    def forward(self, k, q, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            mask = mask.float()
            attn_mask = 1 - torch.bmm(1 - mask, 1 - mask.permute(0, 2, 1))
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask.bool(), -float("inf"))
        # prepend the softmax input with a column of 0
        attn = torch.cat([torch.zeros(attn.shape[0], self.n_site, self.n_comp, 1).to(q.device), attn], 3)
        attn = self.softmax(attn / self.scale)[:, :, :, 1:]
        # row-wise dot product 
        out_v = self.layer_v(attn.contiguous().view(
                -1, self.n_site * self.n_comp, self.n_comp)
                ).view(-1, self.n_site, self.n_comp)
        output = out_v
        return output, attn
    
class AttnLayer(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_comp, n_head, d_x, d_h=0):
        super().__init__()
        self.n_head = n_head
        self.d_x = d_x
        self.d_h = d_h
        self.d_q = max(d_h // 2, d_x * 2)
        
        if self.d_h == 0:
            self.w_q = torch.nn.Linear(self.d_x, n_head * self.d_q)
            self.w_k = torch.nn.Linear(self.d_x, n_head * self.d_q)
        else:
            self.w_q1 = torch.nn.Linear(self.d_x, self.d_h)
            self.w_q2 = torch.nn.Linear(self.d_h, n_head * self.d_q)
            self.w_k1 = torch.nn.Linear(self.d_x, self.d_h)
            self.w_k2 = torch.nn.Linear(self.d_h, n_head * self.d_q)
            self.activ = torch.nn.ELU()

        self.attnblock = AdsorptiveAttnBlock(n_comp, scale=self.d_q ** 0.5, n_site=n_head)


    def forward(self, x, mask=None):
        batchsize, n_comp, _ = x.size()
        if self.d_h == 0:
            q = self.w_q(x)
            k = self.w_k(x)
        else:
            q = self.w_q2(self.activ(self.w_q1(x)))
            k = self.w_k2(self.activ(self.w_k1(x)))

        q = q.view(batchsize, n_comp, self.n_head, self.d_q)
        k = k.view(batchsize, n_comp, self.n_head, self.d_q)

        q, k = q.transpose(1, 2), k.transpose(1, 2)
        output, attn = self.attnblock(k, q, mask)

        return torch.mean(output, 1), attn
        
class SorbNetX(torch.nn.Module):
    
    def __init__(self, n_comp, n_state_each, d_vech=0, n_site=1, **kwargs):
        super().__init__()
        self.d_state = n_state_each
        self.d_vech = d_vech
        self.n_comp = n_comp
        self.n_site = n_site

        self.attn = AttnLayer(n_comp, n_site, n_state_each, d_vech)
        self.activation = torch.nn.functional.elu

    def forward(self, x):
        mask = get_mask(x)
        output, attn = self.attn(x, mask=mask)
        output = output.masked_fill(mask.squeeze(-1).bool(), float("nan"))
        output = torch.clamp(output, EPS, 1 - EPS)
        return output, attn, mask.squeeze(-1)

    def get_value_matrices(self):
        return self.attn.attnblock.layer_v.weight


class MLP(torch.nn.Module):
    
    def __init__(self, n_comp, nz, **kwargs):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_comp + 1, nz), 
            torch.nn.ELU(), 
            torch.nn.Linear(nz, nz),
            torch.nn.ELU(),
            torch.nn.Linear(nz, nz),
            torch.nn.ELU(),
            torch.nn.Linear(nz, n_comp),
            torch.nn.Softplus(),
        )
        
    def __call__(self, x):
        mask = get_mask(x)
        x = torch.cat([x[:, :, 0], x[:, 0:1, 1]], 1)
        return self.model(x), None, mask.squeeze(-1)

model_dict = {
    "SorbNetX": SorbNetX,
    "MLP": MLP,
}

