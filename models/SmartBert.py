import torch 
import torch.nn as nn
from .Classifier import Classifier
from .Gate import Gate
import torch.nn.functional as F
class SmartBert(nn.Module):
    def __init__(self,model,config,cla_hidden_size=128):
        super(SmartBert,self).__init__()
        self.embedding = model.embeddings
        self.encoder = model.encoder
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.classifier = nn.ModuleList([
            Classifier(config,self.num_labels,i+1) for i in range(config.num_hidden_layers)
        ])
        self.gate = nn.ModuleList([
            Gate(config) for i in range(config.num_hidden_layers)
        ])
        self.cla = cla_hidden_size 
        # cross_layer contrast
        self.contrast = nn.ModuleList([nn.Sequential(
            nn.Linear(config.hidden_size,self.cla),
            nn.LayerNorm(self.cla),
            nn.Tanh(),
            nn.Linear(self.cla,self.cla)
        ) for i in range(config.num_hidden_layers)])
        
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        self.threshold = 0
        self.is_hard_weight = False
        self.skipped_rate = 0.5
        # the parameters of skipping rate
        self.lamada = 0.1
        # the parameters of contrastive learning
        self.eta1 = 0.01
        self.eta2 = 0.5
        self.t1 = 0.5
        self.t2 = 0.6
        self.use_gate = True

    def set_eta1(self,x):
        self.eta1 = x 
    
    def set_eta2(self,x):
        self.eta2 = x
    
    def set_t1(self,x):
        self.t1 = x 
    
    def set_t2(self,x):
        self.t2 = x    
    
    def set_use_gate(self,x):
        self.use_gate = x


    def set_lamada(self,x):
        self.lamada = x 

    def set_skipped_rate(self,x):
        self.skipped_rate = x

    def set_hard_weight_mechanism(self,x):
        self.is_hard_weight = x    

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, is_exit = False, labels=None,train_first_stage=False):

        if input_ids is None:
            raise("Please check your input. Your input_ids is None")
        device = input_ids.device
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            mask = attention_mask[:, None, None, :]

        mask = mask.to(dtype=next(self.parameters()).dtype)  
        mask = (1.0 - mask) * -10000.0
        embedding_output = self.embedding(input_ids, token_type_ids, position_ids)
        if self.training:
            if train_first_stage:
                hidden = embedding_output 
                gate_outputs = 0
                pre_hidden_list = []
                for i,layer_module in enumerate(self.encoder.layer):
                    pre_hidden = hidden
                    gate_output = self.gate[i](pre_hidden,mask)
                    gate_outputs += torch.mean(gate_output.squeeze(-1))  
                                      
                    if self.is_hard_weight:
                    # hard weight mechanism
                        gate_output = gate_output + (torch.ge(gate_output,self.skipped_rate).float() - gate_output).detach()

                    layer_outputs = layer_module(pre_hidden, mask)
                    hidden = layer_outputs[0] * (1 - gate_output[:,:,None]) + pre_hidden * gate_output[:,:,None]
                    #
                    pre_hidden_list.append(pre_hidden)
                    #

                logits = self.classifier[-1](hidden,mask)

                loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1)) + self.lamada/gate_outputs
                
                contrast_loss = 0.0
                func = lambda x: torch.exp(x/self.t1)
                for i in range(len(pre_hidden_list) - 1):
                    q1 = F.normalize(self.contrast[i](pre_hidden_list[i]),dim = -1)
                    q2 = F.normalize(self.contrast[i+1](pre_hidden_list[i+1]),dim = -1)
                    token_matric = func(torch.bmm(q1,q2.transpose(-2, -1)))
                    pos = torch.diagonal(token_matric, offset=0, dim1=-1)
                    neg = torch.sum(token_matric,dim=-1)
                    temp = -torch.log(pos/neg)
                    contrast_loss += temp.mean()
                loss = (1-self.eta1) * loss + self.eta1 * contrast_loss
                return loss, logits
            else:
                loss, hidden, hidden_list = 0, embedding_output, []
                with torch.no_grad():
                    for i,layer_module in enumerate(self.encoder.layer):
                        pre_hidden = hidden
                        gate_output = self.gate[i](pre_hidden,mask)
                        if self.is_hard_weight:
                        # hard weight mechanism
                            gate_output = gate_output + (torch.ge(gate_output,self.skipped_rate).float() - gate_output).detach()
                        layer_outputs = layer_module(pre_hidden, mask)
                        hidden = layer_outputs[0] * (1 - gate_output[:,:,None]) + pre_hidden * gate_output[:,:,None]
                        # hidden_list.append(hidden)
                        hidden_list.append(layer_outputs[0])
                    last_logits = self.classifier[-1](hidden_list[-1],mask).view(-1, self.num_labels)
                contrast_output_list = []
                for i in range(self.num_layers - 1):
                    current_logits,contrast_output = self.classifier[i](hidden_list[i], mask, True)
                    loss += self.criterion(current_logits,labels)
                    contrast_output_list.append(contrast_output)
                func = lambda x: torch.exp(x/self.t2)
                contrastive_loss = 0.0
                for i in range(len(contrast_output_list)-2):
                    t1 = F.normalize(contrast_output_list[i],dim=-1)
                    t2 = F.normalize(contrast_output_list[i+1],dim=-1)
                    output = func(torch.bmm(t1,t2.transpose(-2,-1)))
                    pos = torch.diagonal(output,offset=0,dim1=-1)
                    neg = output.sum(-1) - pos
                    temp = -torch.log(pos/neg)                    
                    contrastive_loss += temp.mean()
                loss = (1-self.eta2) * loss + self.eta2 * contrastive_loss/(self.num_layers-2)
                return loss, last_logits
        else:
            # inference 
            if is_exit:
                hidden = embedding_output
                skiped_layer_index = []
                exited_layer_index = []
                index = -1
                for i in range(self.num_layers):
                    index = i
                    if self.use_gate:
                        if self.gate[i](hidden,mask) >= self.skipped_rate:
                            skiped_layer_index.append(i+1)
                            continue


                    layer_outputs = self.encoder.layer[i](hidden, mask)
                    hidden = layer_outputs[0]
                    logits = self.classifier[i](hidden, mask)
                    probs = nn.Softmax(dim=1)(logits)
                    entropy = torch.distributions.Categorical(probs=probs).entropy()
                    if entropy<self.threshold:
                        break
                # logger.info("skiped_layer:{}".format(skiped_layer_index))
                exited_layer_index.append(index+1)
                if index == self.num_layers - 1:
                    logits = self.classifier[-1](hidden, mask)
                return logits,skiped_layer_index,exited_layer_index

            else:
                hidden = embedding_output
                skiped_layer_index = []
                for i in range(self.num_layers):
                    if self.gate[i](hidden,mask) >= self.skipped_rate:
                        skiped_layer_index.append(i+1)
                        continue
                    layer_outputs = self.encoder.layer[i](hidden, mask)
                    hidden = layer_outputs[0]
                logits = self.classifier[-1](hidden,mask)
                # logger.info("skiped_layer:{}".format(skiped_layer_index))
                return logits,skiped_layer_index

    def _difficult_samples_idxs(self, idxs, logits):
        # logits: (batch_size, labels_num)
        probs = nn.Softmax(dim=1)(logits)
        entropys = torch.distributions.Categorical(probs=probs).entropy()
        rel_diff_idxs = (entropys > self.threshold).nonzero().view(-1)
        abs_diff_idxs = torch.tensor([idxs[i] for i in rel_diff_idxs], device=logits.device)
        return abs_diff_idxs, rel_diff_idxs