import torch
import math
import torch.nn as nn
import torch.nn.functional as F


# First Model
class AttentionNetwork(nn.Module):
    def __init__(self, L=512, D=128, n_classes=1):
        super(AttentionNetwork, self).__init__()
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh(),
                            nn.Dropout(0.2)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(0.2)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class CoxPH(nn.Module):
    def __init__(self, input_size, n_classes, dropout):
        super(CoxPH, self).__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            AttentionNetwork(L=256, D=128, n_classes=1)
        )

        self.output = nn.Sequential(
            nn.Linear(256, 24),
            nn.LayerNorm(24),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, n_classes)
        )

        
    def forward(self, x):
        A, h = self.attention_net(x.squeeze(0))   
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)
        return self.output(M)


# Second Model
class Model2(nn.Module):
    def __init__(self, input_size, n_classes, dropout):
        super(Model2, self).__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            AttentionNetwork(L=256, D=128, n_classes=1)
        )

        self.output = nn.Sequential(
            nn.Linear(256, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, n_classes)
        )


    def forward(self, x):
        A, h = self.attention_net(x.squeeze(0))   
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)
        return self.output(M)
    

# Third Model
class MultiHeadAttentionNetwork(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, n_classes=1):
        super(MultiHeadAttentionNetwork, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)
        self.attention_c = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        A = self.attention_c(attn_output)
        return A, attn_output


class ModelAtt(nn.Module):
    def __init__(self, input_size, n_classes, dropout):
        super(ModelAtt, self).__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            MultiHeadAttentionNetwork(embed_dim=256, num_heads=4, n_classes=1)
        )

        self.output = nn.Sequential(
            nn.Linear(256, 24),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, n_classes)
        )

    def forward(self, x):
        total_output = None
        all_A = []
        for tensor in torch.split(x, 1000, dim=0):
            A, h = self.attention_net(tensor)
            A = torch.transpose(A, 1, 0)
            A = F.softmax(A, dim=1)
            
            M = torch.mm(torch.transpose(A.squeeze(1), 1, 0), h.squeeze(0))
            all_A.append(A)

            if total_output is None:
                total_output = M
            else:
                total_output += M
        return M, self.output(total_output), torch.cat(all_A, dim=1)
    
# Fourth Model
class EasyModel(nn.Module):
    def __init__(self, input_size, n_classes, dropout=0.2):
        super(EasyModel, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.AlphaDropout(dropout)
        )

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.f2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        batch_size, seq_length, input_size = x.size()
        x = x.view(batch_size * seq_length, input_size)
        x = self.f1(x)
        x_att, _ = self.attention(x,x,x)
        output = self.f2(x_att)

        output = output.view(batch_size, seq_length, -1)
        return output.mean(dim=1)


# Sixth Model
class Model3(nn.Module):
    def __init__(self, input_size, dropout):
        super(Model3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU()
        )

        self.attention_net = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.2, batch_first=True)

        self.layer3 = nn.Sequential(
            nn.Linear(64, 4),
            nn.LayerNorm(4),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        x = self.layer1(x)
        x = self.layer2(x)

        #mask_numeric = mask.float().cpu()
        #attn_mask_numeric = torch.matmul(mask_numeric, torch.transpose(mask_numeric, 1, 2))
        #attn_mask = attn_mask_numeric.bool().cuda(device=0)

        
        #A, h = self.attention_net(x, x, x, key_padding_mask=mask.squeeze(-1), attn_mask=attn_mask)
        attention = []
        for i in range(x.shape[0]):
            A, h = self.attention_net(x[i, :mask[i], :], x[i, :mask[i], :], x[i, :mask[i], :])
            attention.append(torch.mean(A, dim=0, keepdim=True))
        
        A_final = torch.cat(attention, dim=0)
        output = self.layer3(A_final)

        return output


# Seventh Model
class MilAttention(nn.Module):
    """
    Mil Attention Mechanism
    This layer contains Mil Attention Mechanism
    # Input Shape
        2D tensor with shape: (batch_size, input_dim)
    # Output Shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, L_dim, use_bias=True, use_gated=False):
        super(MilAttention, self).__init__()
        self.L_dim = L_dim
        self.use_bias = use_bias
        self.use_gated = use_gated
        
        self.V = nn.Linear(L_dim, 258)
        self.w = nn.Linear(258, 1)
        
        if use_gated:
            self.U = nn.Linear(L_dim, 1)
        else:
            self.U = None

    def forward(self, x):
        ori_x = x
        
        # V * tanh(x)
        x = torch.tanh(self.V(x))

        if self.use_gated:
            gate_x = torch.sigmoid(self.U(ori_x))
            ac_x = x * gate_x
        else:
            ac_x = x
        
        soft_x = self.w(ac_x)
        alpha = F.softmax(soft_x, dim=0)
        
        return alpha.T

class LastSigmoid(nn.Module):
    """
    Attention Activation
    This layer contains a FC layer which only has one neuron with sigmoid activation
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    """
    def __init__(self, L_dim, output_dim, use_bias=False):
        super(LastSigmoid, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.kernel = nn.Linear(L_dim, output_dim)
        if self.use_bias:
            self.bias = nn.Linear(L_dim, output_dim)
        else:
            self.bias = None

    def activate(self, x):
        a = torch.exp(x[:, 0].unsqueeze(1))
        b = F.softplus(x[:, 1].unsqueeze(1))
        return torch.cat((a, b), dim=1)

    def forward(self, x):
        x = torch.sum(x, dim=0, keepdim=True)

        x = self.kernel(x)
        if self.use_bias:
            x = x + self.bias
        
        out = self.activate(x)
        
        return out


class ModelFinal(nn.Module):
    def __init__(self, input_size, n_classes):
        super(ModelFinal, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.attention = MilAttention(L_dim=input_size, use_gated=False)
        self.last_sigmoid = LastSigmoid(L_dim=input_size, output_dim=n_classes)

    def forward(self, x):
        x = x.squeeze(0)
        alpha = self.attention(x)
        x_mul = torch.matmul(alpha, x)
        out = self.last_sigmoid(x_mul)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        mu = self.linear2(x)
        logvar = self.linear3(x)
        sigma = torch.exp(logvar)

        z = mu + sigma*self.N.sample(mu.shape)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl


class Decoder(nn.Module):
    def __init__(self, input_size, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, input_size)

    def forward(self, z):
        z = F.gelu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VAE(nn.Module):
    def __init__(self, input_size, latent_dims, surv_nodes):
        super(VAE, self).__init__()
        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=1, dropout=0.5, batch_first=True)

        self.encoder = Encoder(input_size, latent_dims)
        self.decoder = Decoder(input_size, latent_dims)

        self.surv_dims = surv_nodes
        self.surv_layers = nn.ModuleList()
        self.dropout_coxnet = nn.Dropout(0.5)


        num_surv_layers = len(self.surv_dims) - 1
        for i in range(num_surv_layers):
            self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))

    
    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            #x_concat = nn.LayerNorm(x_concat.size()[1]).cuda()(x_concat)
            x_concat = nn.Tanh()(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat)

        risk = torch.exp(x_concat)
        return risk


    def forward(self, x, mask):
        attention = []
        for i in range(x.shape[0]):
            A, _ = self.attention_net(x[i, :mask[i], :], x[i, :mask[i], :], x[i, :mask[i], :])
            attention.append(torch.mean(A, dim=0, keepdim=True))
        
        x_att = torch.cat(attention, dim=0)

        z, kl = self.encoder(x_att)

        risk = self.coxnet(z)
        rec = self.decoder(z)
        
        return rec, x_att, risk, kl
    

# Winner
class ModelSurv(nn.Module):
    def __init__(self, input_size, dropout, surv_nodes):
        super(ModelSurv, self).__init__()
        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=2, dropout=dropout, batch_first=True)

        self.surv_dims = surv_nodes
        self.surv_layers = nn.ModuleList()
        self.dropout_coxnet = nn.Dropout(dropout)

        for i in range(len(self.surv_dims) - 1):
            self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))



    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            x_concat = nn.LeakyReLU()(x_concat)
            #x_concat = nn.Tanh()(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat)

        risk = torch.exp(x_concat)
        return risk


    def forward(self, x, mask, explainability=False):
        attention = []

        if explainability:
            A, _ = self.attention_net(x[:mask, :], x[:mask, :], x[:mask, :], need_weights=False)
            risk = self.coxnet(A)
            return risk, A
        else:
            for i in range(x.shape[0]):
                A, _ = self.attention_net(x[i, :mask[i], :], x[i, :mask[i], :], x[i, :mask[i], :], need_weights=False)
                attention.append(torch.mean(A, dim=0, keepdim=True))

                del A
                torch.cuda.empty_cache()

        x_att = torch.cat(attention, dim=0)

        risk = self.coxnet(x_att)

        return risk


class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(MultiHeadAttentionCustom, self).__init__()
        self.nh = num_heads
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3)
        self.Wo = nn.Linear(hidden_size, hidden_size)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)


    def forward(self, x):
        B = 1
        S, C = x.shape
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, S, C)
        return self.out_drop(self.Wo(x))


class MultiOmicsX(nn.Module):
    def __init__(self, input_size, surv_nodes, dropout):
        super(MultiOmicsX, self).__init__()

        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=3, dropout=dropout, batch_first=True)

        self.surv_dims = surv_nodes
        self.surv_layers = nn.ModuleList()
        self.dropout_coxnet = nn.Dropout(dropout)

        for i in range(len(self.surv_dims) - 1):
            self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))



    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            x_concat = nn.LeakyReLU()(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat)

        return x_concat


    def forward(self, x, mask, demog, clin, genomic, transcr):
        attention = []

        for i in range(x.shape[0]):
            A, _ = self.attention_net(x[i, :mask[i], :], x[i, :mask[i], :], x[i, :mask[i], :])
            attention.append(torch.mean(A, dim=0, keepdim=True))

        x_att = torch.cat(attention, dim=0)
        risk = self.coxnet(x_att)

        return risk


class MultiOmicsDF(nn.Module):
    def __init__(self, input_size, dropout, surv_nodes=[512, 32], demo_size=2, clin_size=4, geno_size=40, transcr_size=2021):
        super(MultiOmicsDF, self).__init__()

        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=3, dropout=dropout, batch_first=True)

        self.demo_fc = nn.Sequential(
            nn.Linear(demo_size, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64)
        )
        self.clin_fc = nn.Sequential(
            nn.Linear(clin_size, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64)
        )
        self.geno_fc = nn.Sequential(
            nn.Linear(geno_size, 64),
            nn.Tanh(),
            nn.BatchNorm1d(64)
        )
        self.transcr_fc = nn.Sequential(
            nn.Linear(transcr_size, 128),
            nn.SELU(),
            nn.AlphaDropout(dropout),
        )

        self.deep_fusion = nn.Sequential(
            nn.Linear(1536 + 64 + 64 + 64 + 128, 1024),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.surv_dims = surv_nodes
        self.surv_layers = nn.ModuleList()
        self.dropout_coxnet = nn.Dropout(dropout)
        for i in range(len(self.surv_dims) - 1):
            self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))
        
        self.final_layer = nn.Linear(self.surv_dims[-1], 1)

    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            x_concat = nn.LeakyReLU()(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat)

        return x_concat

    def forward(self, x, mask, demog, clin, genomic, transcr):
        attention = []
        for i in range(x.shape[0]):
            A, _ = self.attention_net(x[i, :int(mask[i]), :], x[i, :int(mask[i]), :], x[i, :int(mask[i]), :], need_weights=False)
            attention.append(torch.mean(A, dim=0, keepdim=True))

            del A
            torch.cuda.empty_cache()
        
        x = torch.cat(attention, dim=0)
        del attention

        demog = self.demo_fc(demog)
        clin = self.clin_fc(clin)
        genomic = self.geno_fc(genomic)
        transcr = self.transcr_fc(transcr)

        x = torch.cat([x, demog, clin, genomic, transcr], dim=1)
        x = self.deep_fusion(x)

        risk = self.coxnet(x)

        del x, demog, clin, genomic, transcr
        torch.cuda.empty_cache()

        risk = self.final_layer(risk)
        return torch.exp(risk)
    


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, dropout):
        super(MixtureOfExperts, self).__init__()

        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=3, dropout=dropout, batch_first=True)

        self.image_expert = Expert(1536, 128)
        self.demog_expert = Expert(2, 128)
        self.clin_expert = Expert(4, 128)
        self.genomic_expert = Expert(40, 128)
        self.transcr_expert = Expert(2021, 128)
        
        self.gating_network = GatingNetwork(3603, 5)

        self.final_layer = nn.Linear(128, 1)
    
    def forward(self, x_img, masking, x_demog, x_clin, x_genomic, x_transcr):
        attention = []
        for i in range(x_img.shape[0]):
            A, _ = self.attention_net(x_img[i, :int(masking[i]), :], 
                                      x_img[i, :int(masking[i]), :], 
                                      x_img[i, :int(masking[i]), :], need_weights=False)
            attention.append(torch.mean(A, dim=0, keepdim=True))
            del A
            torch.cuda.empty_cache()
        
        x_img = torch.cat(attention, dim=0)
        del attention, masking

        x_concat = torch.cat([x_img, x_demog, x_clin, x_genomic, x_transcr], dim=1)
        weights = self.gating_network(x_concat).unsqueeze(-1)
        
        x_img = self.image_expert(x_img)
        x_demog = self.demog_expert(x_demog)
        x_clin = self.clin_expert(x_clin)
        x_genomic = self.genomic_expert(x_genomic)
        x_transcr = self.transcr_expert(x_transcr)
        
        experts_output = torch.stack([x_img, x_demog, x_clin, x_genomic, x_transcr], dim=1)
        final_expert_output = (experts_output.view(experts_output.shape[0], 5, -1) * weights).sum(dim=1)
        risk_score = self.final_layer(final_expert_output).squeeze(1)
        
        return torch.exp(risk_score)