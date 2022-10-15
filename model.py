import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class cnn_lstm(nn.Module):
    def __init__(self, feature_extractor, feat_dim=1024, n_layers=1):
        super(cnn_lstm,self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layers = n_layers
        
        feature_extractor = feature_extractor
        d = OrderedDict(feature_extractor.named_children())
        print(d.keys()) # now we have all layers of the feature extractor, we just need to remove the last one.
        _, fc = d.popitem(last=True)

        fe_out_planes = fc.in_features  # this is the input dimention of the fc layer
        
        self.feature_extraction = nn.Sequential(d)
        self.feature_extraction.avgpool = nn.AdaptiveAvgPool2d(1) # here we are updating one layer of the netowrk
        
        self.lstm_fc = nn.LSTM(input_size=fe_out_planes, hidden_size=feat_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.feat_dim*2, 1)
        
        self.to(torch.device('cuda:0'))
   
    def forward(self, x):
         # x : BxTxCxHxW
        s = x.size()
        
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.feat_dim).to(torch.device('cuda:0')) # we have to have a hidden value for each unit of all rnn-layers.
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.feat_dim).to(torch.device('cuda:0'))
        
        x = x.view(-1, *s[2:]) # x: (B*T)x(C)x(H)x(W))
        x = self.feature_extraction(x).squeeze()
        print(x.size()) # (B*T), d)  
        x = x.view(s[0], s[1], -1)  # x: BxTxd
        print(x.size())
      
        x, (self.fc_h, self.fc_c) = self.lstm_fc(x, (h0, c0))  # BxTxd'
        print(x.size())
        
        # getting last time from left to right
        x = x[:,-1,:]
        
        out = self.fc(x)
        return out
        
        
        
        
if __name__ == '__main__':
    model = cnn_lstm(feature_extractor=models.resnet18(pretrained=False))
    x = torch.rand(2, 10, 3, 120, 160).to(torch.device('cuda:0'))
    out = model(x)
    print(out.shape)