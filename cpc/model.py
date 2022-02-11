# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio

import torch

###########################################
# Networks
###########################################
class MLP(nn.Module):
    def __init__(self,in_dim, output_dim,hidden_dim=128):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_dim,hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim*2,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
        
    def forward(self,din):
#         din = din.view(-1,28*28)
        dout = F.tanh(self.fc1(din))
#         dout = F.relu(self.fc2(dout))
        return self.fc3(dout)

class IDModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm"):

        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class MFCCEncoder(nn.Module):

    def __init__(self,
                 dimEncoded):

        super(MFCCEncoder, self).__init__()
        melkwargs = {"n_mels": max(128, dimEncoded), "n_fft": 321}
        self.dimEncoded = dimEncoded
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=dimEncoded,
                                               melkwargs=melkwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MFCC(x)
        return x.permute(0, 2, 1)


class LFBEnconder(nn.Module):

    def __init__(self, dimEncoded, normalize=True):

        super(LFBEnconder, self).__init__()
        self.dimEncoded = dimEncoded
        self.conv = nn.Conv1d(1, 2 * dimEncoded,
                              400, stride=1)
        self.register_buffer('han', torch.hann_window(400).view(1, 1, 400))
        self.instancenorm = nn.InstanceNorm1d(dimEncoded, momentum=1) \
            if normalize else None

    def forward(self, x):

        N, C, L = x.size()
        x = self.conv(x)
        x = x.view(N, self.dimEncoded, 2, -1)
        x = x[:, :, 0, :]**2 + x[:, :, 1, :]**2
        x = x.view(N * self.dimEncoded, 1,  -1)
        x = torch.nn.functional.conv1d(x, self.han, bias=None,
                                       stride=160, padding=350)
        x = x.view(N, self.dimEncoded,  -1)
        x = torch.log(1 + torch.abs(x))

        # Normalization
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                                   num_layers=nLevelsGRU, batch_first=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)
        else:
            self.baseNet = nn.GRU(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x, h
    def rnn_forward(self, x, h):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, h)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x, h


class NoAr(nn.Module):

    def __init__(self, *args):
        super(NoAr, self).__init__()

    def forward(self, x):
        return x


class BiDIRARTangled(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRARTangled, self).__init__()
        assert(dimOutput % 2 == 0)

        self.ARNet = nn.GRU(dimEncoded, dimOutput // 2,
                            num_layers=nLevelsGRU, batch_first=True,
                            bidirectional=True)

    def getDimOutput(self):
        return self.ARNet.hidden_size * 2

    def forward(self, x):

        self.ARNet.flatten_parameters()
        xf, _ = self.ARNet(x)
        return xf


class BiDIRAR(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRAR, self).__init__()
        assert(dimOutput % 2 == 0)

        self.netForward = nn.GRU(dimEncoded, dimOutput // 2,
                                 num_layers=nLevelsGRU, batch_first=True)
        self.netBackward = nn.GRU(dimEncoded, dimOutput // 2,
                                  num_layers=nLevelsGRU, batch_first=True)

    def getDimOutput(self):
        return self.netForward.hidden_size * 2

    def forward(self, x):

        self.netForward.flatten_parameters()
        self.netBackward.flatten_parameters()
        xf, _ = self.netForward(x)
        xb, _ = self.netBackward(torch.flip(x, [1]))
        return torch.cat([xf, torch.flip(xb, [1])], dim=2)


###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,_ = self.gAR(encodedData)
        return cFeature, encodedData, label
    
class CPCModel_t(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel_t, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label
    
class CPCModel_Promote_t(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128):

        super(CPCModel_Promote_t, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        params = torch.ones([1, 1, hidden_dim], requires_grad=True)
        self.promote = nn.Parameter(params)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(torch.cat([encodedData,
                                       self.promote.repeat([batch_size,1,1])],dim=1))
        return cFeature[:,-1], cFeature[:,:-1], encodedData, label
    
class CPCModel_Promote(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128):

        super(CPCModel_Promote, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        params = torch.ones([1, 1, hidden_dim], requires_grad=True)
        self.promote = nn.Parameter(params)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,_ = self.gAR(torch.cat([encodedData,
                                       self.promote.repeat([batch_size,1,1])],dim=1))
        return cFeature[:,-1], cFeature[:,:-1], encodedData, label
    
class CPCModel_Promote_multi(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128,
                 re_output_dim=4):

        super(CPCModel_Promote_multi, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR 
        self.prompt_list = torch.nn.ParameterList()
        for i in range(re_output_dim):
            self.prompt_list.append(nn.Parameter(torch.randn([1, 1, hidden_dim], requires_grad=True)))
        
        # self.projection_list=torch.nn.ModuleList()
        # for i in range(hidden_dim//pe_output_dim):
        #     self.projection_list.append(nn.Linear(hidden_dim,pe_output_dim))

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,h = self.gAR(encodedData)
        x_list=[]
        for i,prompt in enumerate(self.prompt_list):
            x_,_=self.gAR.rnn_forward(prompt.repeat([batch_size,1,1]),h)
            # x_=self.projection_list[i](x_)
            x_list.append(x_)
        represent=torch.cat(x_list,dim=-2)
        return represent, cFeature, encodedData, label
    
class CPCModel_Promote_multi_seq(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128,
                 prompt_seq=4):

        super(CPCModel_Promote_multi_seq, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR 
        self.prompt=nn.Parameter(torch.ones([1, prompt_seq, hidden_dim], requires_grad=True))
        
        # self.projection_list=torch.nn.ModuleList()
        # for i in range(hidden_dim//pe_output_dim):
        #     self.projection_list.append(nn.Linear(hidden_dim,pe_output_dim))

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,h = self.gAR(encodedData)
        represent,_=self.gAR.rnn_forward(self.prompt.repeat([batch_size,1,1]),h)
        return represent, cFeature, encodedData, label
    
class CPCModel_Promote_multi_v2(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128,
                 pe_output_dim=4):

        super(CPCModel_Promote_multi_v2, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR 
        self.prompt_list = torch.nn.ParameterList()
        for i in range(hidden_dim//pe_output_dim):
            self.prompt_list.append(nn.Parameter(torch.ones([1, 1, hidden_dim], requires_grad=True)))
        
        self.projection_list=torch.nn.ModuleList()
        for i in range(hidden_dim//pe_output_dim):
            self.projection_list.append(nn.Linear(hidden_dim,pe_output_dim))

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,h = self.gAR(encodedData)
        x_list=[]
        for i,prompt in enumerate(self.prompt_list):
            x_,_=self.gAR.rnn_forward(prompt.repeat([batch_size,1,1]),h)
            x_=self.projection_list[i](x_)
            x_list.append(x_)
        represent=torch.cat(x_list,dim=-2)
        return represent, cFeature, encodedData, label
    
class CPCModel_Promote_multi_v3(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128,
                 pe_output_dim=4):

        super(CPCModel_Promote_multi_v3, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR 
        self.prompt_list = torch.nn.ParameterList()
        for i in range(hidden_dim//pe_output_dim):
            self.prompt_list.append(nn.Parameter(torch.ones([1, 1, hidden_dim], requires_grad=True)))
        
        self.projection=nn.Linear(hidden_dim,pe_output_dim)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,h = self.gAR(encodedData)
        x_list=[]
        for i,prompt in enumerate(self.prompt_list):
            x_,_=self.gAR.rnn_forward(prompt.repeat([batch_size,1,1]),h)
            x_=self.projection(x_)
            x_list.append(x_)
        represent=torch.cat(x_list,dim=-2)
        return represent, cFeature, encodedData, label
    
class CPCModel_Promote_multi_v3_t(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128,
                 pe_output_dim=4):

        super(CPCModel_Promote_multi_v3_t, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR 
        self.prompt_list = torch.nn.ParameterList()
        for i in range(hidden_dim//pe_output_dim):
            self.prompt_list.append(nn.Parameter(torch.ones([1, 1, hidden_dim], requires_grad=True)))
        
        self.projection=nn.Linear(hidden_dim,pe_output_dim)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,h = self.gAR(encodedData)
        x_list=[]
        for i,prompt in enumerate(self.prompt_list):
            x_=self.gAR(torch.cat([encodedData,
                                   promote.repeat([batch_size,1,1])],dim=1))
            x_=self.projection(x_)
            x_list.append(x_)
        represent=torch.cat(x_list,dim=-2)
        return represent, cFeature, encodedData, label
    
class CPCModel_Promote_multi_seq_t(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128,
                 prompt_seq=4):

        super(CPCModel_Promote_multi_seq_t, self).__init__()
        self.prompt_seq=prompt_seq
        self.gEncoder = encoder
        self.gAR = AR 
        self.prompt=nn.Parameter(torch.randn([1, prompt_seq, hidden_dim], requires_grad=True))
        
        self.projection=nn.Linear(hidden_dim,hidden_dim//prompt_seq)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(torch.cat([encodedData,
                                       self.prompt.repeat([batch_size,1,1])],dim=1))
        # represent = self.projection(cFeature[:,-self.prompt_seq:])
        represent = cFeature[:,-self.prompt_seq:]
        
        return represent, cFeature[:,:-self.prompt_seq], encodedData, label
    
class CPCModel_Promote_v1(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128):

        super(CPCModel_Promote_v1, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        params = torch.ones([1, 1, hidden_dim], requires_grad=True)
        self.promote = nn.Parameter(params)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        encodedData = torch.cat([encodedData,
                                 self.promote.repeat([batch_size,1,1])],dim=1)
        cFeature,_ = self.gAR(encodedData)
        return cFeature[:,-1], cFeature, encodedData, label
    
class CPCModel_Promote_v2(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128):

        super(CPCModel_Promote_v2, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        params = torch.ones([1, 1, hidden_dim], requires_grad=True)
        self.promote = nn.Parameter(params)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature,_ = self.gAR(torch.cat([encodedData,
                                       encodedData.mean(1,keepdim=True)+\
                                       self.promote.repeat([batch_size,1,1])],dim=1))
        return cFeature[:,-1], cFeature[:,:-1], encodedData, label
class CPCModel_Promote_v3(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 seq_len=19,
                 hidden_dim=128):

        super(CPCModel_Promote_v3, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        params = torch.ones([1, 1, hidden_dim], requires_grad=True)
        self.promote_layer = MLP(hidden_dim,hidden_dim,hidden_dim)

    def forward(self, batchData, label):
        batch_size = batchData.size()[0]
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        encodedData = torch.cat([encodedData,
                                 self.promote_layer(encodedData.mean(1,keepdim=True))],dim=1)
        cFeature,_ = self.gAR(encodedData)
        return cFeature[:,-1], cFeature, encodedData, label

class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):

        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):

        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData, label = model(batchData, label)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), \
            torch.cat(outEncoded, dim=2), label
