import torch 
import torch.nn as nn
from .MultiheadAttention import MultiheadAttention
class Classifier(nn.Module):
    def __init__(self,config,num_labels,cur_layer,cla_heads_num=4, cla_hidden_size=128):
        super(Classifier,self).__init__()
        self.input_size = config.hidden_size
        self.num_labels = num_labels
        self.cla_heads_num = cla_heads_num
        self.cla_hidden_size = cla_hidden_size
        self.output_layer_0 = nn.Linear(self.input_size, self.cla_hidden_size)
        self.atten = MultiheadAttention(self.cla_hidden_size, self.cla_heads_num, config.attention_probs_dropout_prob)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, self.num_labels)             
        self.main = nn.Sequential(
            nn.Linear(self.cla_hidden_size,self.cla_hidden_size),
            nn.LayerNorm(self.cla_hidden_size),
            nn.Tanh(),
            nn.Linear(self.cla_hidden_size,self.cla_hidden_size)
        )


    def forward(self, hidden, mask, is_contrast = False):
        hidden = torch.tanh(self.output_layer_0(hidden))
        atten_output = self.atten(hidden, mask)
        if is_contrast:
            attn_output = self.main(hidden)
        hidden = atten_output[0]
        hidden = hidden[:,0,:]
        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        if is_contrast:
            return logits,attn_output
        return logits