import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, input_type):
        super(ConvBlock,self).__init__()
        if input_type == 'image':
            self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
            self.add_module('norm',nn.BatchNorm2d(out_channel)),
        else:
            self.add_module('conv', nn.Conv1d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
            # self.add_module('conv',nn.Conv1d(in_channel, out_channel, kernel_size=ker_size, stride=2, padding=padd)),
            self.add_module('norm', nn.BatchNorm1d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt.input_type)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt.input_type)
            self.body.add_module('block%d'%(i+1),block)
        if opt.input_type == 'image':
            self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
        else:
            self.tail = nn.Conv1d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
            # self.tail = nn.Conv1d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=2, padding=opt.padd_size)

    def forward(self,x):
        # print("@ WDiscriminator1: x.shape=", x.shape)
        x = self.head(x)
        # print("@ WDiscriminator2: x.shape=",x.shape)
        x = self.body(x)
        # print("@ WDiscriminator3: x.shape=", x.shape)
        x = self.tail(x)
        # print("@ WDiscriminator4: x.shape=", x.shape)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1,opt.input_type) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        self.input_type = opt.input_type
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1,opt.input_type)
            self.body.add_module('block%d'%(i+1),block)
        if opt.input_type == 'image':
            self.tail = nn.Sequential(
                nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
                nn.Tanh()
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv1d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
                # nn.Conv1d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=2, padding=opt.padd_size),
                nn.Tanh()
            )
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        if self.input_type == 'image':
            y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        else:
            y = y[:, :, ind:(y.shape[2] - ind)]
        return x+y
