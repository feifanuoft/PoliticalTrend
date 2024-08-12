import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import args_parameter

args = args_parameter.args

class CNNTextClassifier(nn.Module):
  def __init__(self, vocab,args):
    self.k1 = args.k1   
    self.k2 = args.k2 
    self.n1 = args.n1 
    self.n2 = args.n2
    self.freeze_embedding = args.freeze_embedding
    self.bias = args.bias
    super(CNNTextClassifier,self).__init__()
    self.emdedding_size = vocab.vectors.shape[1] 
    self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=self.freeze_embedding)
    self.cnn_layer1 = torch.nn.Conv2d(1,out_channels = self.n1,kernel_size=(self.k1,self.emdedding_size),bias=self.bias)
    self.cnn_layer2 = torch.nn.Conv2d(1,out_channels = self.n2,kernel_size=(self.k2,self.emdedding_size),bias=self.bias)
    self.activation = torch.nn.ReLU()
    self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    self.fc = torch.nn.Linear(self.n1+self.n2,1)

  def forward(self, x):
    length = len(x)
    embedded_result = self.embedding(x)
    embedded_result = torch.transpose(embedded_result, 0, 1).unsqueeze(1)
    cnn_layer1= self.cnn_layer1(embedded_result)
    cnn_layer2 = self.cnn_layer2(embedded_result)
    layer1_active = self.activation(cnn_layer1)
    layer2_active = self.activation(cnn_layer2)
    # print(layer1_active.size())
    # print(layer2_active.size())
    max_pool_1= self.maxpool(layer1_active)
    max_pool_2= self.maxpool(layer2_active)
    concat_result = torch.cat((max_pool_1,max_pool_2),1)
    # print(concat_result.size())
    prediction = self.fc(concat_result.squeeze())
    return prediction.squeeze()
