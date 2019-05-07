import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# utility function to return different ResNets
def get_resnet_model(model=None, use_pretrained=True):
    if model is None:
        resnet = models.resnet152(pretrained=use_pretrained)
    elif model.lower() == "resnet18":
        resnet = models.resnet18(pretrained=use_pretrained)
    elif model.lower() == "resnet34":
        resnet = models.resnet34(pretrained=use_pretrained)
    elif model.lower() == "resnet50":
        resnet = models.resnet50(pretrained=use_pretrained)
    elif model.lower() == "resnet101":
        resnet = models.resnet101(pretrained=use_pretrained)
    elif model.lower() == "resnet152":
        resnet = models.resnet152(pretrained=use_pretrained)
    else:
        print("{} is not a valid version for ResNet".format(model))
        sys.exit(1)
    return resnet

class LRCN(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_classes, bidirectional=False,
                 model=None, use_pretrained=True, feature_extracting=True):
        super(LRCN, self).__init__()
        # base CNN
        resnet = get_resnet_model(model, use_pretrained)
        base_modules = list(resnet.children())[:-1] # remove the last linear layer
        self.resnet = nn.Sequential(*base_modules)
        # freeze all params of base CNN if feature extracting mode selected
        if feature_extracting:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # RNN for sequential processing
        self.rnn = nn.LSTM(
            input_size=list(self.resnet.parameters())[-1].size(0),  # size of the last avgpool layer
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional)
        # final linear layer for classification
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, batch_seqs):
        batch_size, time_step, c, h, w = batch_seqs.shape
        batch_out = torch.stack([self.resnet(mip_seq) for mip_seq in batch_seqs])
        # cnn_out shape (batch_size, seq_length/time_step, flattened_mip_features)
        cnn_out = batch_out.reshape(batch_size, time_step, -1)
        # rnn_out has the same shape as cnn_out
        # h_n has shape (n_layers, batch_size, hidden_size)
        # h_c has shape (n_layers, batch_size, hidden_size)
        rnn_out, (h_n, h_c) = self.rnn(cnn_out, None)   # None is used to set zero initial hidden state
        out = self.fc(rnn_out[:, -1, :])                # pass the rnn_out from the last time step to the classifier, i.e., (many-to-one)
        return out
