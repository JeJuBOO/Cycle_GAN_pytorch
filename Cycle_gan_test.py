import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools

from model import *
from dataset import *

# test
# HyperParameters
lr = 0.0002
batch_size = 2
image_size = 256
in_channels_img = 3
out_channels_img = 3
num_epoch = 200
wgt_ident = 0.5
wgt_cycle = 10

features_d = 64
features_g = 64

transform_test = transforms.Compose([ Normalize(), ToTensor()])
transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

dataset_test = Dataset("C:/cloud/OneDrive/Learning_Data/horse2zebra/horse2zebra/",transform=transform_test,data_type='both')
loader_train = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

num_train = len(loader_train)
num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Networks
netG_A2B = Generator(in_channels_img, out_channels_img).to(device)
netG_B2A = Generator(out_channels_img, in_channels_img).to(device)
netD_A = Discriminator(in_channels_img).to(device)
netD_B = Discriminator(out_channels_img).to(device)

initialize_weights(netG_A2B)
initialize_weights(netG_B2A)
initialize_weights(netD_A)
initialize_weights(netD_B)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(0.5, 0.999))


# Lossess
cri_GAN = torch.nn.BCEWithLogitsLoss().to(device)
cri_cycle = torch.nn.L1Loss().to(device)
cri_identity = torch.nn.L1Loss().to(device)

writer_test_image = SummaryWriter(f'./runs/horse2zebra/test_10/image')
writer_test_loss = SummaryWriter(f'./runs/horse2zebra/test_10/loss')

netG_A2B.load_state_dict(torch.load('./Save_model/netG_A2B.pt'))
netG_B2A.load_state_dict(torch.load('./Save_model/netG_B2A.pt'))
netD_A.load_state_dict(torch.load('./Save_model/netD_A.pt'))
netD_B.load_state_dict(torch.load('./Save_model/netD_B.pt'))
with torch.no_grad():
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    loss_G_a2b_test = []
    loss_G_b2a_test = []
    loss_D_a_test = []
    loss_D_b_test = []
    loss_Cycle_a_test = []
    loss_Cycle_b_test = []
    loss_Iden_a_test = []
    loss_Iden_b_test = []
    for i, test in enumerate(loader_train, 1):
        
        input_A = test['dataA'].to(device)
        input_B = test['dataB'].to(device)
        output_B = netG_A2B(input_A)

        output_B = netG_A2B(input_A)
        reout_A = netG_B2A(output_B)
        
        output_A = netG_B2A(input_B)
        reout_B = netG_A2B(output_A)
        
        pred_real_A = netD_A(input_A)
        pred_fake_A = netD_A(output_A.detach())
        loss_D_real_A = cri_GAN(pred_real_A, torch.ones_like(pred_real_A))
        loss_D_fake_A = cri_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = (loss_D_real_A + loss_D_fake_A)*5.0
        
        pred_real_B = netD_B(input_B)
        pred_fake_B = netD_B(output_B.detach())
        loss_D_real_B = cri_GAN(pred_real_B, torch.ones_like(pred_real_B))
        loss_D_fake_B = cri_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = (loss_D_real_B + loss_D_fake_B)*5.0
            
        loss_D = loss_D_A + loss_D_B
        
        pred_fake_A = netD_A(output_A)
        pred_fake_B = netD_B(output_B)
        
        loss_G_A2B = cri_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_G_B2A = cri_GAN(pred_fake_B, torch.ones_like(pred_fake_A))
        
        loss_Cycle_A = cri_cycle(reout_A,input_A)
        loss_Cycle_B = cri_cycle(reout_B,input_B)
            
        identity_A = netG_B2A(input_A)
        identity_B = netG_B2A(input_B)
        loss_identity_A = cri_identity(input_A, identity_A)
        loss_identity_B = cri_identity(input_B, identity_B)
        
        loss_G = (loss_G_A2B + loss_G_B2A) + \
            wgt_cycle*(loss_Cycle_A + loss_Cycle_B) + \
            wgt_ident*(loss_identity_A + loss_identity_B)
                
        
        loss_D_a_test += [loss_D_A.item()]
        loss_D_b_test += [loss_D_B.item()]
        loss_G_a2b_test += [loss_G_A2B.item()]
        loss_G_b2a_test += [loss_G_B2A.item()]
        loss_Cycle_a_test += [loss_Cycle_A.item()]
        loss_Cycle_b_test += [loss_Cycle_B.item()]
        loss_Iden_a_test += [loss_identity_A.item()]
        loss_Iden_b_test += [loss_identity_B.item()]
        
        input_A = transform_inv(input_A)
        input_B = transform_inv(input_B)
        output_A = transform_inv(output_A)
        output_B = transform_inv(output_B)
        reout_A = transform_inv(reout_A)
        reout_B = transform_inv(reout_B)
        
        writer_test_image.add_images('input_a', input_A, i, dataformats='NHWC')
        writer_test_image.add_images('output_a', output_A, i, dataformats='NHWC')
        writer_test_image.add_images('input_b', input_B,  i, dataformats='NHWC')
        writer_test_image.add_images('output_b', output_B,  i, dataformats='NHWC')
        writer_test_image.add_images('reout_a', reout_A,  i, dataformats='NHWC')
        writer_test_image.add_images('reout_b', reout_B,  i, dataformats='NHWC')
        
print(f' Batch : {i}\
            GAN   A2B : {np.mean(loss_G_a2b_test):.4f}, B2A : {np.mean(loss_G_b2a_test):.4f} \
            DISC  A   : {np.mean(loss_D_a_test):.4f} B : {np.mean(loss_D_b_test):.4f}, \
            Cycle A   : {np.mean(loss_Cycle_a_test):.4f} B : {np.mean(loss_Cycle_b_test):.4f}, \
            Ident A   : {np.mean(loss_Iden_a_test):.4f} B : {np.mean(loss_Iden_b_test):.4f}')