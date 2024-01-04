import torch 
import torch.nn as nn
from .MultiheadAttention import MultiheadAttention
class Gate(nn.Module):
    def __init__(self,config,cla_hidden_size=128,cla_heads_num=2):
        super(Gate,self).__init__()
        self.input_size = config.hidden_size
        self.num_label = 1
        self.cla_hidden_size = cla_hidden_size
        self.cla_heads_num = cla_heads_num
        self.output_layer_0 = nn.Linear(self.input_size, self.cla_hidden_size)
        self.atten = MultiheadAttention(self.cla_hidden_size, self.cla_heads_num, config.attention_probs_dropout_prob)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.num_label)

    def forward(self, hidden, mask):
        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.atten(hidden,mask)[0]
        hidden = hidden[:,0,:]
        logits = self.output_layer_1(hidden)
        probs = torch.sigmoid(logits)
        return probs