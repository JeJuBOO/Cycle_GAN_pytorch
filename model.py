from layer import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, features_g=64):
        super(Generator, self).__init__()

        self.encoder1   = Encoder(in_ch          , features_g     , kernel_size=7 , stride=1 , reflection_pad=(7-1)//2)
        self.encoder2   = Encoder(features_g     , features_g * 2 , kernel_size=3 , stride=2 , reflection_pad=(3-1)//2)
        self.encoder3   = Encoder(features_g * 2 , features_g * 4 , kernel_size=3 , stride=2 , reflection_pad=(3-1)//2)
        self.res_block1 = Resbolck(features_g * 4, kernel_size=3)
        self.decoder1   = Decoder(features_g * 4 , features_g * 2 , kernel_size=3 , stride=2 , padding=(3-1)//2 , output_padding=1)
        self.decoder2   = Decoder(features_g * 2 , features_g     , kernel_size=3 , stride=2 , padding=(3-1)//2 , output_padding=1)
        self.out        = Output(features_g      , out_ch         , kernel_size=7 , reflection_pad=(7-1)//2 )
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        res_b1 = self.res_block1(enc3)
        res_b2 = self.res_block1(res_b1)
        res_b3 = self.res_block1(res_b2)
        res_b4 = self.res_block1(res_b3)
        res_b5 = self.res_block1(res_b4)
        res_b6 = self.res_block1(res_b5)
        dec1 = self.decoder1(res_b6)
        dec2 = self.decoder2(dec1)
        out = self.out(dec2)
        
        x = torch.tanh(out)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, in_ch,features_d = 64, norm="Inorm"):
        super(Discriminator, self).__init__()
        
        self.disc1 = patchGan(in_ch  , features_d    ,kernel_size=4,stride=2,padding=1,norm=None,relu=0.2)
        self.disc2 = patchGan(features_d    , features_d * 2,kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2)
        self.disc3 = patchGan(features_d * 2, features_d * 4,kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2)
        self.disc4 = patchGan(features_d * 4, features_d * 8,kernel_size=4,stride=1,padding=1,norm=norm,relu=0.2)
        self.disc5 = patchGan(features_d * 8, 1             ,kernel_size=4,stride=1,padding=1,norm=None,relu=None)
        

        
    def forward(self, x): # mini * 6   * 256  * 256
        x = self.disc1(x) # mini * 64  * 128  * 128
        x = self.disc2(x) # mini * 128 * 64  * 64
        x = self.disc3(x) # mini * 256 * 32  * 32
        x = self.disc4(x) # mini * 512 * 31  * 31
        x = self.disc5(x) # mini * 1   * 30  * 30
        
        # Average pooling and flatten
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad