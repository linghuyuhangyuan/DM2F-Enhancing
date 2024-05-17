# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from model import DM2FNet, DM2FNet_woPhy
from tools.config import OHAZE_ROOT
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset, OHazeDataset
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from networks import VGGLoss, LocalEnhancer, weights_init
from unet import RefinementNet, MultiColor_Fusion, MultiColor_direct_fusion

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class Laplacian(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        G = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float()
        # G = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).float()
        G = G.unsqueeze(0).unsqueeze(0)
        G = torch.cat([G, G, G], 0)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        return x
    
def cal_hflf_loss(gt, coarse, pred_res, criterion, laplacian):
    gt_res = gt - coarse
    hf_pred = laplacian(pred_res)
    lf_pred = pred_res - laplacian(pred_res)
    hf_gt = laplacian(gt_res)
    lf_gt = gt_res - laplacian(gt_res)
    
    return criterion(hf_pred, hf_gt) + criterion(lf_pred, lf_gt)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='/backup/zlt/DM2F-Net/ckpts/ckpt', help='checkpoint path')
    parser.add_argument('--load-dm2f', default='/backup/zlt/DM2F-Net/ckpts/ckpt/O-Haze/iter_20000_loss_0.04937_lr_0.000000.pth', help='pretrained dm2f path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args

cfgs = {
    'use_physical': True,
    'iter_num': 40000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': 'iter_20000_loss_0.04937_lr_0.000000',
    'val_freq': 2000,
    'crop_size': 512
}

if __name__ == '__main__':
    args = parse_args()
    
    print(torch.cuda.device_count())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = OHazeDataset(OHAZE_ROOT, 'train_crop_512')
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)
    
    criterion = nn.L1Loss().cuda()
    criterionVGG = VGGLoss(gpu_ids=args.gpus).cuda()
    laplacian = Laplacian().cuda()
    
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')
    
    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')
    
    net = DM2FNet_woPhy().cuda()
    # net_refine = LocalEnhancer(input_nc=3, output_nc=3).cuda()
    # net_refine.apply(weights_init)
    # net_refine = MultiColor_direct_fusion().cuda()
    net_refine = MultiColor_Fusion().cuda()
    # net_refine.load_state_dict(torch.load("/backup/zlt/DM2F-Net/ckpts/ckpt_ohaze_refine_res_lapall_colorfusion/RESIDE_ITS/iter_26000_loss_0.97038_lr_0.000194.pth"))
    
    ckpt_path = '/backup/zlt/DM2F-Net/ckpts/ckpt'
    # exp_name = 'RESIDE_ITS'
    exp_name = 'O-Haze'

    net.load_state_dict(torch.load(args.load_dm2f))
    net.eval()
    
    optimizer = optim.Adam([
        {'params': [param for name, param in net_refine.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net_refine.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])
    
    #### training
    curr_iter = cfgs['last_iter']

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_L1_record = AvgMeter()
        loss_VGG_record = AvgMeter()
        loss_L1_final_record = AvgMeter()
        loss_L1_rgb_record = AvgMeter()
        loss_L1_hsv_record = AvgMeter()
        loss_L1_lab_record = AvgMeter()
        
        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
    
            haze, gt, _ = data
            
            batch_size = haze.size(0)
            
            haze = haze.cuda()
            gt = gt.cuda()
            device = haze.device
            
            optimizer.zero_grad()
            
            coarse_result = sliding_forward(net, haze, device=device).detach()
            # coarse_result = net(haze).detach()
            # net_refine.rgb_to_lab(coarse_result)
            # net_refine.rgb_to_hsv(coarse_result)
            # exit(0)
            out, rgb_f, lab_f, hsv_f = net_refine(coarse_result, haze)
            # out = net_refine(coarse_result, haze)
            
            loss_L1_final = cal_hflf_loss(gt=gt, 
                                          coarse=coarse_result, 
                                          pred_res=out, 
                                          criterion=criterion,
                                          laplacian=laplacian)
            loss_L1_rgb = cal_hflf_loss(gt=gt, 
                                        coarse=coarse_result, 
                                        pred_res=rgb_f, 
                                        criterion=criterion,
                                        laplacian=laplacian)
            loss_L1_lab = cal_hflf_loss(gt=net_refine.rgb_to_lab(gt), 
                                        coarse=net_refine.rgb_to_lab(coarse_result), 
                                        pred_res=lab_f, 
                                        criterion=criterion,
                                        laplacian=laplacian)
            loss_L1_hsv = cal_hflf_loss(gt=net_refine.rgb_to_hsv(gt), 
                                        coarse=net_refine.rgb_to_hsv(coarse_result), 
                                        pred_res=hsv_f, 
                                        criterion=criterion,
                                        laplacian=laplacian)
            # loss_VGG = 0
            loss_L1 = loss_L1_final * 3 + loss_L1_rgb + loss_L1_lab +loss_L1_hsv
            loss = loss_L1
            loss.backward()
            
            optimizer.step()
            
            # update recorder
            train_loss_record.update(loss.item(), batch_size)
            loss_L1_record.update(loss_L1.item(), batch_size)
            loss_L1_final_record.update(loss_L1_final.item(), batch_size)
            loss_L1_rgb_record.update(loss_L1_rgb.item(), batch_size)
            loss_L1_lab_record.update(loss_L1_lab.item(), batch_size)
            loss_L1_hsv_record.update(loss_L1_hsv.item(), batch_size)
            
            curr_iter += 1
            
            log = '[iter %d], [train loss %.5f], [loss_L1_final %.5f], [loss_L1_rgb %.5f], [loss_L1_lab %.5f], [loss_L1_hsv %.5f] ' \
                  '[lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_L1_final_record.avg, loss_L1_rgb_record.avg, loss_L1_lab_record.avg, loss_L1_hsv_record.avg,  optimizer.param_groups[1]['lr'])
            
            # log = '[iter %d], [train loss %.5f], [loss_L1_final %.5f] ' \
            #       '[lr %.13f]' % \
            #       (curr_iter, train_loss_record.avg, loss_L1_final_record.avg,  optimizer.param_groups[1]['lr'])
                  
            print(log)
            open(log_path, 'a').write(log + '\n')
            
            if (curr_iter + 1) % cfgs['val_freq'] == 0:
            #     validate(net, curr_iter, optimizer)
                snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, train_loss_record.avg, optimizer.param_groups[1]['lr'])
                torch.save(net_refine.state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
                torch.save(optimizer.state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))
            
            if curr_iter > cfgs['iter_num']:
                break
            
            
    
    
    