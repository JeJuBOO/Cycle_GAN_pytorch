from model import *
from dataset import *

import torch

import itertools

from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# HyperParameters
lr = 0.0002
batch_size = 2
image_size = 256
in_channels_img = 3
out_channels_img = 3
num_epoch = 200
wgt_ident = 0.5
wgt_cycle = 10

# Dataset
transform_train = transforms.Compose([Normalize(), RandomFlip(), Rescale((image_size+30, image_size+30)), RandomCrop((image_size, image_size)), ToTensor()])
transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

dataset_train =  Dataset("C:/cloud/OneDrive/Learning_Data/horse2zebra/horse2zebra/",
                      data_type='both',transform=transform_train)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_train = len(dataset_train)
num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

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

writer_train_image = SummaryWriter(f'runs/horse2zebra/train/image')
writer_train_loss = SummaryWriter(f'runs/horse2zebra/train/loss')
for epoch in range(num_epoch):
    # training
    netG_A2B.train()
    netG_A2B.train()
    netD_A.train()
    netD_B.train()

    loss_G_a2b_train = []
    loss_G_b2a_train = []
    loss_D_a_train = []
    loss_D_b_train = []
    loss_Cycle_a_train = []
    loss_Cycle_b_train = []
    loss_Iden_a_train = []
    loss_Iden_b_train = []
    
    for i, batch in enumerate(loader_train, 1):
        # model input dataset
        input_A = batch['dataA'].to(device)
        input_B = batch['dataB'].to(device)

        output_B = netG_A2B(input_A)
        reout_A = netG_B2A(output_B)
        
        output_A = netG_B2A(input_B)
        reout_B = netG_A2B(output_A)
        
        set_requires_grad([netD_A, netD_B], True)
        optimizer_D.zero_grad()
        
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
        loss_D.backward()
        optimizer_D.step()
        
        set_requires_grad([netD_A, netD_B], False)
        optimizer_G.zero_grad()
        
        
        pred_fake_A = netD_A(output_A)
        pred_fake_B = netD_B(output_B)
        
        loss_G_A2B = cri_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_G_B2A = cri_GAN(pred_fake_B, torch.ones_like(pred_fake_A))
        
        loss_Cycle_A = cri_cycle(reout_A,input_A)
        loss_Cycle_B = cri_cycle(reout_B,input_B)
        
        
        identity_A = netG_B2A(input_A)
        identity_B = netG_A2B(input_B)
        loss_identity_A = cri_identity(input_A, identity_A)
        loss_identity_B = cri_identity(input_B, identity_B)
        
        loss_G = (loss_G_A2B + loss_G_B2A) + \
            wgt_cycle*(loss_Cycle_A + loss_Cycle_B) + \
            wgt_ident*(loss_identity_A + loss_identity_B)
            
        loss_G.backward()
        optimizer_G.step()
        
        
        loss_D_a_train += [loss_D_A.item()]
        loss_D_b_train += [loss_D_B.item()]
        loss_G_a2b_train += [loss_G_A2B.item()]
        loss_G_b2a_train += [loss_G_B2A.item()]
        loss_Cycle_a_train += [loss_Cycle_A.item()]
        loss_Cycle_b_train += [loss_Cycle_B.item()]
        loss_Iden_a_train += [loss_identity_A.item()]
        loss_Iden_b_train += [loss_identity_B.item()]
        
        if i % 50 == 0:
            input_A = transform_inv(input_A)
            output_A = transform_inv(output_A)
            input_B = transform_inv(input_B)
            output_B = transform_inv(output_B)
            reout_A = transform_inv(reout_A)
            reout_B = transform_inv(reout_B)

            writer_train_image.add_images('input_A', input_A, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            writer_train_image.add_images('output_B', output_B, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            writer_train_image.add_images('reconstruct_A', reout_A, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            
            writer_train_image.add_images('input_B', input_B, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            writer_train_image.add_images('output_A', output_A, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            writer_train_image.add_images('reconstruct_B', reout_B, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            print(f'Epoch : {epoch} | Batch : {i}\
                GAN   A2B : {np.mean(loss_G_a2b_train):.4f}, B2A : {np.mean(loss_G_b2a_train):.4f} \
                DISC  A   : {np.mean(loss_D_a_train):.4f} B : {np.mean(loss_D_b_train):.4f}, \
                Cycle A   : {np.mean(loss_Cycle_a_train):.4f} B : {np.mean(loss_Cycle_b_train):.4f}, \
                Ident A   : {np.mean(loss_Iden_a_train):.4f} B : {np.mean(loss_Iden_b_train):.4f}')
            
        
    writer_train_loss.add_scalar('Gan A2B loss train', np.mean(loss_G_a2b_train), epoch)
    writer_train_loss.add_scalar('Gan B2A loss train', np.mean(loss_G_b2a_train), epoch)
    writer_train_loss.add_scalar('Disc A loss train', np.mean(loss_D_a_train), epoch)
    writer_train_loss.add_scalar('Disc B loss train', np.mean(loss_D_b_train), epoch)
    writer_train_loss.add_scalar('Cycle A loss train', np.mean(loss_Cycle_a_train), epoch)
    writer_train_loss.add_scalar('Cycle B loss train', np.mean(loss_Cycle_b_train), epoch)
    writer_train_loss.add_scalar('Ident A loss train', np.mean(loss_Iden_a_train), epoch)
    writer_train_loss.add_scalar('Ident B loss train', np.mean(loss_Iden_b_train), epoch)

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'Save_Model/netG_A2B.pt')
    torch.save(netG_B2A.state_dict(), 'Save_Model/netG_B2A.pt')
    torch.save(netD_A.state_dict(), 'Save_Model/netD_A.pt')
    torch.save(netD_B.state_dict(), 'Save_Model/netD_B.pt')
    
writer_train_loss.close()
writer_train_image.close()  

