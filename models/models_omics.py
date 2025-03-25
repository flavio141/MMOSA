import torch
import torch.nn as nn


class MultiOmicsModel(nn.Module):
    def __init__(self, input_size, dropout, surv_nodes, transc_nodes):
        super(MultiOmicsModel, self).__init__()
        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=3, dropout=dropout, batch_first=True)

        self.surv_dims = surv_nodes
        self.surv_layers = nn.ModuleList()
        self.dropout_coxnet = nn.Dropout(dropout)

        for i in range(len(self.surv_dims) - 1):
            self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))

        self.transc_dims = transc_nodes
        self.transc_layers = nn.ModuleList()
        self.dropout_transcr = nn.Dropout(dropout)

        for j in range(len(self.transc_dims) - 1):
            self.transc_layers.append(nn.Linear(self.transc_dims[j], self.transc_dims[j+1]))


        self.output = nn.Sequential(
            nn.Linear(self.surv_dims[-1] + self.transc_dims[-1] + 46, 1)
        )

    def transcr(self, x_transcr):
        for j, layer in enumerate(self.transc_layers):
            x_transcr = layer(x_transcr)
            x_transcr = nn.LeakyReLU()(x_transcr)
            x_transcr = nn.Dropout(0.2)(x_transcr)
            if j != len(self.transc_layers) - 1:
                x_transcr = self.dropout_transcr(x_transcr)

        return x_transcr


    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            x_concat = nn.Tanh()(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat)

        return x_concat


    def forward(self, x, mask, demog, clin, genomic, transcr):
        attention = []

        for i in range(x.shape[0]):
            A, _ = self.attention_net(x[i, :mask[i], :], x[i, :mask[i], :], x[i, :mask[i], :], need_weights=False)
            attention.append(torch.mean(A, dim=0, keepdim=True))
            
            A = A.detach()
            del A
            torch.cuda.empty_cache()

        x_att = torch.cat(attention, dim=0)

        x_images = self.coxnet(x_att)
        x_transcr = self.transcr(transcr)
        x_dcg = torch.cat([demog, clin, genomic], dim=1)
        
        x_final = torch.cat([x_dcg, x_transcr, x_images], dim=1)
        output = self.output(x_final)
        risk = torch.exp(output)
        return risk
    

# Late Integration
class ModelImages(nn.Module):
    def __init__(self, input_size, dropout, surv_nodes):
        super(ModelImages, self).__init__()
        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=3, dropout=dropout, batch_first=True)

        self.surv_dims = surv_nodes
        self.surv_layers = nn.ModuleList()
        self.dropout_coxnet = nn.Dropout(dropout)

        for i in range(len(self.surv_dims) - 1):
            self.surv_layers.append(nn.Linear(self.surv_dims[i], self.surv_dims[i+1]))



    def coxnet(self, x_concat):
        for i, layer in enumerate(self.surv_layers):
            x_concat = layer(x_concat)
            if i == len(self.surv_layers) - 2:
                x_repr = x_concat
            x_concat = nn.LeakyReLU()(x_concat)
            if i != len(self.surv_layers) - 1:
                x_concat = self.dropout_coxnet(x_concat)

        risk = torch.exp(x_concat)
        return risk, x_repr

    def forward(self, x, mask):
        attention = []

        for i in range(x.shape[0]):
            A, _ = self.attention_net(x[i, :mask[i], :], x[i, :mask[i], :], x[i, :mask[i], :], need_weights=False)
            attention.append(torch.median(A, dim=0, keepdim=True).values)

            A = A.detach()
            del A
            torch.cuda.empty_cache()

        x_att = torch.cat(attention, dim=0)
        risk, x_repr = self.coxnet(x_att)
        return risk, x_repr
    


class OtherOmicsModel(nn.Module):
    def __init__(self, input_size, dropout, transc_nodes):
        super(OtherOmicsModel, self).__init__()
        self.attention_net = nn.MultiheadAttention(embed_dim=input_size, num_heads=1, dropout=dropout, batch_first=True)

        self.transc_dims = transc_nodes
        self.transc_layers = nn.ModuleList()
        self.dropout_transcr = nn.Dropout(dropout)

        for j in range(len(self.transc_dims) - 1):
            self.transc_layers.append(nn.Linear(self.transc_dims[j], self.transc_dims[j+1]))

    def transcr(self, x_transcr):
        for j, layer in enumerate(self.transc_layers):
            x_transcr = layer(x_transcr)
            x_transcr = nn.ELU()(x_transcr)
            x_transcr = nn.Dropout(0.2)(x_transcr)
            if j != len(self.transc_layers) - 1:
                x_transcr = self.dropout_transcr(x_transcr)
        
        risk = torch.exp(x_transcr)
        return risk

    def forward(self, transcr, genomic):
        x_att, _ = self.attention_net(transcr, transcr, transcr, need_weights=False)

        x = torch.cat([x_att, genomic], dim=1)
        risk = self.transcr(x)
        return risk