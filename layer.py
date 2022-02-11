import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,in_ch,out_ch=64,kernel_size=7,stride=1,padding=0,reflection_pad=None):
        super().__init__()
        layer = []
        
        if reflection_pad != None:
            layer += [nn.ReflectionPad2d(reflection_pad)]
        
        layer += [  nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,padding=padding),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True) ]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)

class Resbolck(nn.Module):
    def __init__(self,in_ch,kernel_size=3):
        super().__init__()
        
        layer = [   nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size),
                    nn.InstanceNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size),
                    nn.InstanceNorm2d(in_ch)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=2,padding=1,output_padding=1):
        super().__init__()
        
        layer = [   nn.ConvTranspose2d(in_ch, out_ch,kernel_size=kernel_size,
                                    stride=stride,padding=padding,output_padding=output_padding),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True) ]

        self.net = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.net(x)
    
class Output(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=7,reflection_pad=3):
        super().__init__()

        layer = [   nn.ReflectionPad2d(reflection_pad),
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)

class patchGan(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4,stride=1,padding=1,norm='Inorm',relu=None,dropout=None):
        super().__init__()
        
        layer = []
        layer += [nn.Conv2d(in_ch, out_ch,kernel_size=kernel_size,stride=stride,padding=padding)]
        
        if norm != None:
            layer += [nn.InstanceNorm2d(out_ch)]
            
        if relu != None:
            layer += [nn.LeakyReLU(relu)]

        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)