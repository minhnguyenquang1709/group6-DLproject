import numpy
import os
import argparse
import copy
import math 
import utils
import torch
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import SRTransG as models
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.autograd import Variable
from datetime import datetime, date
import matplotlib.cm as mpl_color_map
import cv2

# run after training

def get_concat_h(im):
    #print('im',len(im))
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
        _psnr, _ssim = utils.calc_psnr_and_ssim(real[imagesinbatch].detach(), gen[imagesinbatch].detach())
        psnr += _psnr
        ssim += _ssim
    psnr=psnr/gen.shape[0]
    ssim=ssim/gen.shape[0]         
    return psnr,ssim

def mapdataset(generator,datasetloc):
    generator.eval()
    datasetname=os.path.basename(datasetloc)
    i=0
    data_loader,tot_batch = utils.get_dataloader(datasetloc,img_size=args.img_size, batch_size = args.batch_size)
    for data in tqdm(range(tot_batch)):
        realB4 = next(data_loader).to(device)
        realB2 = nn.AvgPool2d(2, stride=2)(realB4)
        realA = nn.AvgPool2d(2, stride=2)(realB2)
        real_A = Variable(realA).cuda()
        
        A= real_A.requires_grad_()
        fake_B = generator.forward(A)
        temp = torch.ones(1, 3, fake_B.shape[3], fake_B.shape[3], dtype=torch.float, requires_grad=True, device='cuda')
        
        fake_B.backward(realB2)
        saliency, _ = torch.max(A.grad.data.abs(), dim=1)
        saliency=saliency-saliency.min()
        saliency=saliency/saliency.max()
        saliency=saliency.squeeze()
        
        imageA = A.squeeze()
        imageA=(imageA+1)/2
        
        plt.subplot(121)
        plt.imshow(imageA.cpu().detach().numpy().transpose(1,2,0))
        plt.subplot(122)
        plt.imshow(saliency.cpu(),cmap='jet')
        plt.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.savefig(str(args.weight)+'/saliency/'+datasetname+str(i)+'a.png')
        plt.close('all')
        i=i+1
        
        
def map(generator):
    _logger=utils.Logger(log_file_name=os.path.join(str(args.weight),'test.log'), 
        logger_name='testlog').get_log()
    _logger.info("test module")
    generator.eval()
    mapdataset(generator,'../dataset/DIV2K_valid_HR' )
    mapdataset(generator,'../dataset/Set5' )
    mapdataset(generator,'../dataset/Set14' )
    mapdataset(generator,'../dataset/BSD100' )
    mapdataset(generator,'../dataset/urban100' )
    mapdataset(generator,'../dataset/manga' )
    generator.train()
    return 


def testdataset(generator,datasetloc,_logger):
    datasetname=os.path.basename(datasetloc)
    data_loader,tot_batch = utils.get_dataloader(datasetloc,img_size=args.img_size, batch_size = args.batch_size)
    psnr=ssim=psnr2=ssim2=0
    i=0
    with torch.no_grad():
        for data in tqdm(range(tot_batch)):
            #images, labels = data
            # calculate outputs by running images through the network
            realB4 = next(data_loader).to(device)
            realB2 = nn.AvgPool2d(2, stride=2)(realB4)
            realA = nn.AvgPool2d(2, stride=2)(realB2)
            realAup2 = nn.Upsample(scale_factor=2)(realB2)
            realAup4 = nn.Upsample(scale_factor=4)(realA)
            
            gen2 = generator(realA)
            gen4 = generator(gen2)
            gen2=generator(realB2)
            # the class with the highest energy is what we choose as prediction
            psnr_,ssim_=calpsnrssim(gen2,realB4)
            psnr2_,ssim2_=calpsnrssim(gen4,realB4)
            #psnr2_,ssim2_=calpsnrssim(gen4,realB4)
            psnr+=psnr_
            psnr2+=psnr2_
            ssim+=ssim_
            ssim2+=ssim2_
            #map(generator,realB2)
            
            nrow=int(math.sqrt( realA.shape[0]))
            realimg = T.ToPILImage()(make_grid(realA, nrow = nrow, padding = 10, normalize = True))
            groundth2 = T.ToPILImage()(make_grid(realB2, nrow = nrow, padding = 10, normalize = True))
            groundth4 = T.ToPILImage()(make_grid(realB4, nrow = nrow, padding = 10, normalize = True))
            genimg2 = T.ToPILImage()(make_grid(gen2, nrow = nrow, padding = 10, normalize = True))
            genimg4 = T.ToPILImage()(make_grid(gen4, nrow = nrow, padding = 10, normalize = True))
            get_concat_h([realimg,genimg4,groundth4]).save(str(args.weight)+'/test_results/'+datasetname+str(i)+'x4.jpg')
            get_concat_h([groundth2,genimg2,groundth4]).save(str(args.weight)+'/test_results/'+datasetname+str(i)+'x2.jpg')
            
            grounddiff4 = T.ToPILImage()(make_grid(realB4-realAup4, nrow = nrow, padding = 10, normalize = True))
            genrateddiff4 = T.ToPILImage()(make_grid(gen4-realAup4, nrow = nrow, padding = 10, normalize = True))
            grounddiff2 = T.ToPILImage()(make_grid(realB4-realAup2, nrow = nrow, padding = 10, normalize = True))
            genrateddiff2 = T.ToPILImage()(make_grid(gen2-realAup2, nrow = nrow, padding = 10, normalize = True))
            get_concat_h([grounddiff4,genrateddiff4]).save(str(args.weight)+'/diff/'+datasetname+str(i)+'x4.jpg')
            get_concat_h([grounddiff2,genrateddiff2]).save(str(args.weight)+'/diff/'+datasetname+str(i)+'x2.jpg')
            
            x=realB4-realAup4
            y= gen4-realAup4
            #print(x.shape,torch.min(x),torch.max(x))
            #print(y.shape,torch.min(y),torch.max(y))
            
            
            #print(grounddiff4.shape,grounddiff4.min,grounddiff4.max)
            #print(genrateddiff4.shape,genrateddiff4.min,genrateddiff4.max)
            
            #break
            i=i+1
            
    psnr=psnr/tot_batch
    psnr2=psnr2/tot_batch
    ssim=ssim/tot_batch
    ssim2=ssim2/tot_batch
    _logger.info(datasetname+': x2psnr/ssim: %.3f' %psnr+'/%.3f' %ssim+' x4psnr/ssim: %.3f' %psnr2+'/%.3f' %ssim2)
    print('psnrx2,ssimx2,psnr4,ssim4',psnr,ssim,psnr2,ssim2)
    
def test(generator):
    _logger=utils.Logger(log_file_name=os.path.join(str(args.weight),'test.log'), 
        logger_name='testlog').get_log()
    _logger.info("")
    _logger.info("Test module:  "+ datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    generator.eval()
    #testdataset(generator,'/home/galvin/code/dataset/DIV2K_valid_HR',_logger )
    testdataset(generator,'../dataset/Set5',_logger )
    testdataset(generator,'../dataset/Set14',_logger )
    testdataset(generator,'../dataset/BSD100',_logger )
    testdataset(generator,'../dataset/urban100',_logger )
    testdataset(generator,'../dataset/manga',_logger )
    generator.train()
    return 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type = int, default = 1,
                        help = "Size of each batches (Default: 1)")
    parser.add_argument("--beta1", type = float, default = 0.0,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,
                        help = "Coefficients used for computing running averages of gradient and its square")
    # change to the directory of input low-res folder
    parser.add_argument("--data-dir", type = str, default = "./data/div2k/test",
                        help = "Data root dir of test data")
    
    parser.add_argument("--img_size", type = int, default = 512,
                        help = "Select the specific img_size to training")
    parser.add_argument("--weight", type = str, default = "",
                        help = "Select the specific img_size to training")
    args = parser.parse_args()
    # Device
    device = torch.device("cuda:0")
    # Initialize Generator and Discriminator
    netG = nn.DataParallel(models.SRTransG()).to(device)
    netG.load_state_dict(copy.deepcopy(torch.load(str(args.weight)+"/weights/Generator.pth")))   
    # Loss function
    criterion = nn.BCELoss()
    criterionL1 = torch.nn.L1Loss()
    criterionL2 = torch.nn.MSELoss()
    
    # Start Testing
    test(netG)
    #start activation map
    #map(netG)
