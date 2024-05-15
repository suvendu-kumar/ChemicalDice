from ChemicalDice.myImports import *
import torch.nn.init as init
import math

class Autoencoder(nn.Module):
    def __init__(self, dims):
        super(Autoencoder, self).__init__()

        self.dims = dims

        latent_space_ind = 0
        latent_space_dim = 1e5

        for i in range(len(self.dims)):
            if self.dims[i] < latent_space_dim:
                latent_space_dim = self.dims[i]
                latent_space_ind = i

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(self.dims)-1):
            if i < latent_space_ind:
                self.encoder.append(nn.Linear(dims[i], dims[i+1]))
                self.encoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.Linear(dims[i], dims[i+1]))
                self.decoder.append(nn.ReLU())

        # Weight initialization
        self.init_weights()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        for l in self.encoder:
            x = l(x)
        return x
    
    def decode(self, x):
        for l in self.decoder:
            x = l(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
    
class ChemicalDiceIntegrator(nn.Module):
    def getSum(self, lst, ind):
        res = 0
        for i in range(len(lst)):
            if i != ind:
                res += lst[i]
        return res
    
    def getAEDimensions(self, inp_dim, latent_space_dim):
        ae_dim = []
        this_dim = inp_dim
        while this_dim > latent_space_dim:
            ae_dim += [this_dim]
            this_dim = math.ceil(this_dim/3)
        
        return ae_dim + [latent_space_dim] + ae_dim[::-1]
    
    def remove_element_at_index(self, lst, index):
        if index < 0 or index >= len(lst):
            raise IndexError("Index out of range")

        return lst[:index] + lst[index + 1:]
    
    def __init__(self, latent_space_dims, embedding_dim, embd_sizes,embd_sizes_sum, lr=1e-3,weight_decay=0):
        super(ChemicalDiceIntegrator,self).__init__()

        self.latent_space_dims = latent_space_dims
        self.encoders = nn.ModuleDict({})

        # for i in range(6):
        #     print(self.getAEDimensions(self.getSum(self.latent_space_dims, i), self.latent_space_dims[i]))
        # [20006, 19834, 18796, 17018, 10218, 15218]
        self.encoders[f'{0}'] = Autoencoder([embd_sizes_sum[0], 2223, 741, embd_sizes[0], 741, 2223, embd_sizes_sum[0]])
        self.encoders[f'{1}'] = Autoencoder([embd_sizes_sum[1], 2204, embd_sizes[1], 2204, embd_sizes_sum[1]])
        self.encoders[f'{2}'] = Autoencoder([embd_sizes_sum[2], 2089, embd_sizes[2], 2089, embd_sizes_sum[2]])
        self.encoders[f'{3}'] = Autoencoder([embd_sizes_sum[3], 5672, embd_sizes[3], 5672, embd_sizes_sum[3]])
        self.encoders[f'{4}'] = Autoencoder([embd_sizes_sum[4], embd_sizes[4], embd_sizes_sum[4]])
        self.encoders[f'{5}'] = Autoencoder([embd_sizes_sum[5], 5072, embd_sizes[5], 5072, embd_sizes_sum[5]])

        ae_dim = self.getAEDimensions(self.getSum(self.latent_space_dims, -1), embedding_dim)
        # print(ae_dim)
        self.encoders[f'{6}'] = Autoencoder(ae_dim)
        

    def forward(self, x):
        inp = []
        for i in range(len(x)):
            inp += [torch.cat(self.remove_element_at_index(x, i), dim=1)]

        enc = []
        op = []

        for i in range(len(x)):
            enc_n = self.encoders[f'{i}'].encode(inp[i])
            # print(enc_n.shape)
            enc += [self.encoders[f'{i}'].encode(inp[i])]
            op += [self.encoders[f'{i}'](inp[i])]
        
        i=6
        concat_key = torch.cat(enc, dim=1)
        # print(concat_key.shape)
        concat_key_enc = self.encoders[f'{i}'].encode(concat_key)
        concat_key_op = self.encoders[f'{i}'](concat_key)

        return enc[0], enc[1], enc[2], enc[3], enc[4], enc[5], op[0], op[1], op[2], op[3], op[4], op[5], concat_key_enc, concat_key_op

class Classifier(nn.Module):
    def __init__(self, latent_space_dim = 100, lr=1e-3,weight_decay=0):
        super(Classifier,self).__init__()
        self.latent_space_dim = latent_space_dim
        self.classifier = nn.Sequential(
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 2),
        )
        

    def forward(self, x):
        op = self.classifier(x)
        return op
        # return enc_op, conc_op