### SRTransGAN: SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network

import os
import argparse
import copy
from tqdm import tqdm
from utils import mkExpDir
import utils
import SRTransG
import model
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchsummary import summary
from utils import calc_psnr_and_ssim
import math 
from PIL import Image
import time

def get_concat_h(im):
    iw=0
    for i in im:
        iw=iw+i.width
    dst = Image.new('RGB', (iw, im[len(im)-1].height))
    
    iw=0
    for i in im:
        dst.paste(i, (iw, 0))
        iw=iw+i.width
    return dst

def calpsnrssim(gen,real):
    psnr=0
    ssim=0
    for imagesinbatch in  range(gen.shape[0]):
        _psnr, _ssim = calc_psnr_and_ssim(real[imagesinbatch].detach(), gen[imagesinbatch].detach())
        psnr += _psnr
        ssim += _ssim
    psnr=psnr/gen.shape[0]
    ssim=ssim/gen.shape[0]         
    return psnr,ssim

def test(generator):
    data_loader,tot_batch = utils.get_dataloader(args.test_dir,img_size=args.img_size, batch_size = args.batch_size)
    generator.eval()    
    psnr=ssim=psnr2=ssim2=0
    totda=0
    with torch.no_grad():
        for data in range(tot_batch):
            realB4 = next(data_loader).to(device)
            realB2 = nn.AvgPool2d(2, stride=2)(realB4)
            realA = nn.AvgPool2d(2, stride=2)(realB2)
            
            gen2 = generator(realA)
            gen4 = generator(gen2)
            gen2=generator(realB2)
            psnr_,ssim_=calpsnrssim(gen2,realB4)
            psnr2_,ssim2_=calpsnrssim(gen4,realB4)
            
            psnr+=psnr_
            psnr2+=psnr2_
            ssim+=ssim_
            ssim2+=ssim2_
    generator.train()
    psnr=psnr/tot_batch
    ssim=ssim/tot_batch
    psnr2=psnr2/tot_batch
    ssim2=ssim2/tot_batch
    return psnr,psnr2,ssim,ssim2

def train(generator, discriminator, optim_g,optim_d, data_loader, tot_batch, device):
    gen_params = sum(p.numel() for p in generator.parameters())
    dis_params = sum(p.numel() for p in discriminator.parameters())
    _logger = mkExpDir(args.name)
    _logger.info(str(time.asctime(time.localtime())))
    _logger.info("generator.total_params: %d" %gen_params )
    _logger.info("discriminator.total_params: %d" %dis_params )
    _logger.info(args.name)
    _logger.info(generator)
    bestssim=0
    bestno=0
    
    for step in tqdm(range(args.epoch*tot_batch+1)):
        #Ground truth images.
        realB = next(data_loader).to(device)
        realA = nn.AvgPool2d(2, stride=2)(realB)
        realAup = nn.Upsample(scale_factor=2)(realA)
        r_label = torch.ones(args.batch_size).to(device)
        f_label = torch.zeros(args.batch_size).to(device)
        
        #Call Generator 
        fakeB= generator(realA)
        fake_AB = torch.cat((realAup, fakeB), 1)
        real_AB = torch.cat((realAup, realB), 1)
        
        # Train Discriminator1------------------------------------------------------------------------
        ##### require grad for dis true
        requires_grad=True
        if not isinstance(discriminator, list):
            nets = [discriminator]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
        optim_d.zero_grad()
        r_logit = discriminator(real_AB).flatten()
        f_logit = discriminator(fake_AB.detach()).flatten()
        lossD_real = criterion(r_logit, r_label)
        lossD_fake = criterion(f_logit, f_label)
        lossD2=(lossD_real+lossD_fake)*0.5
        lossD2.backward()
        optim_d.step()
        
        # Train Generator1---------------------------------------------------------------------------------
        ##### require grad for dis false
        requires_grad=False
        if not isinstance(discriminator, list):
            nets = [discriminator]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        requires_grad=False
        
        optim_g.zero_grad()
        f_logit = discriminator(fake_AB).flatten()
        lossG = criterion(f_logit, r_label)*0.9
        lossL1=criterionL1(fakeB,realB)*2
        lossL2=criterionL2(fakeB,realB)*.8
        lossL2sr = criterionL2(nn.AvgPool2d(2, stride=2)(fakeB),realA)*.8
        lossGt=lossG+lossL1
        lossGt.backward()
        optim_g.step()

        #training ends----------------------------------------------------------------------------------    
        if step % args.sample_interval == 1:
            psnr,psnr2,ssim,ssim2=test(generator)
            torch.save(generator.state_dict(), args.name+'/weights/Generator_L.pth')
            torch.save(discriminator.state_dict(), args.name+'/weights/Discriminator_L.pth')
            if  bestssim<ssim:
                bestssim=ssim
                bestno =0
                torch.save(generator.state_dict(), args.name+'/weights/Generator.pth')
                # torch.save(discriminator.state_dict(), args.name+'/weights/Discriminator.pth')
            elif bestno>30:
                lr=optim_g.param_groups[0]['lr']*0.2
                optim_g.param_groups[0]['lr']=lr
                _logger.info('updated lr: %.5f' %lr)
                bestno =0
            else:
                bestno+=1
            _logger.info('step: %d' %step +' Gt: %.3f' %(lossGt.item())+' D: %.3f' %(lossD2.item())+ ' G: %.3f' %(lossG.item()) +' L1: %.3f' %(lossL1.item())+' L2: %.3f' %(lossL2.item())+' psnr: %.3f' %psnr+' ssim: %.3f' %ssim+' ssimx4: %.3f' %ssim2 +' bssim: %.3f' %bestssim)
            
            nrow=int(math.sqrt(realA.shape[0]))
            realimg = T.ToPILImage()(make_grid(realA, nrow = nrow, padding = 10, normalize = True))
            genimg = T.ToPILImage()(make_grid(fakeB, nrow = nrow, padding = 10, normalize = True))
            groundth = T.ToPILImage()(make_grid(realB, nrow = nrow, padding = 10, normalize = True))
            get_concat_h([realimg,genimg,groundth]).save(args.name+'/save_results/AB{:05d}.jpg'.format(step))
    psnr,psnr2,ssim,ssim2=test(generator)
    _logger.info('test result: psnrx4: %.3f' %psnr2+' ssimx4: %.3f' %ssim2+' ssim: %.3f' %ssim +' psnr: %.3f' %psnr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type = int, default = 200,
                        help = "Number of epoch for training (Default: 200)")
    parser.add_argument("--batch-size", type = int, default = 1,
                        help = "Size of each batches (Default: 1)")
    parser.add_argument("--lr", type = float, default = 0.0002,
                        help = "Learning rate (Default: 0.0002)")
    parser.add_argument("--beta1", type = float, default = 0.0,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--latent-dim", type = int, default = 1024,
                        help = "Dimension of the latent vector")
    
    # modify the paths in this section to match your dataset directories ------------
    parser.add_argument("--data-dir", type = str, default = "../trying/data/div2k/DIV2K_train_LR_x8/noclass",
                        help = "Data root dir of your training data (low-res images)")
    parser.add_argument("--test-dir", type = str, default = '../trying/data/div2k/DIV2K_train_HR/noclass',
                        help = "Data root dir of your training data (high-res images)")
    # endblock -------------------------------------------------------------------

    parser.add_argument("--sample-interval", type = int, default = 1000,
                        help = "Interval for sampling image from generator")
    parser.add_argument("--gpuid", type = str, default = 0,
                        help = "Select the specific gpu to training")
    parser.add_argument("--img_size", type = int, default = 256,
                        help = "Select the specific img_size to training")
    parser.add_argument("--weight", type = str, default = "",
                        help = "Select the weights to continuing training")
    parser.add_argument("--name", type = str, default = "save_dir",
                        help = "Select the specific name to training")
    args = parser.parse_args()
    
    
    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Dataloader")
    data_loader,tot_batch = utils.get_dataloader(args.data_dir,img_size=args.img_size, batch_size = args.batch_size)
    print('tot_batch',tot_batch)
    
    if args.sample_interval>tot_batch:
         args.sample_interval=tot_batch-1
            
    # Initialize Generator and Discriminator
    netG = nn.DataParallel(SRTransG.SRTransG()).to(device)
    netD = nn.DataParallel(model.Discriminator()).to(device)
      
    if str(args.weight)!="":
        netG.load_state_dict(copy.deepcopy(torch.load(str(args.weight)+"/weights/Generator.pth",device)))    
        netD.load_state_dict(copy.deepcopy(torch.load(str(args.weight)+"/weights/Discriminator.pth",device)))

    # Loss function
    criterion = nn.BCELoss()
    criterionL1 = torch.nn.L1Loss()
    criterionL2 = torch.nn.MSELoss()
    
    # Optimizer and lr_scheduler
    optimizer_g = torch.optim.Adam(netG.parameters(), lr = args.lr,betas = (args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(netD.parameters(), lr = args.lr,betas = (args.beta1, args.beta2))
    
    # Start Training
    train(netG, netD, optimizer_g,optimizer_d, data_loader,tot_batch, device)
