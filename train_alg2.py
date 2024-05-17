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
from networks import define_D, GANLoss
from image_pool import ImagePool

from model import DM2FNet
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset
from tools.utils import AvgMeter, check_mkdir
from networks import VGGLoss

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 20000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_D': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 2000,
    'crop_size': 256
}

def cal_FM(criterion, pred_fake, pred_real, num_D=1, n_layers_D=3, lambda_feat=10.0):
    loss_G_GAN_Feat = 0
    feat_weights = 4.0 / (n_layers_D + 1)
    D_weights = 1.0 / num_D
    for i in range(num_D):
        for j in range(len(pred_fake[i])-1):
            loss_G_GAN_Feat += D_weights * feat_weights * \
                criterion(pred_fake[i][j], pred_real[i][j].detach()) * lambda_feat
    
    return loss_G_GAN_Feat

def main():
    net = DM2FNet(use_refine=True, use_sep=True).cuda().train()
    netD = define_D(input_nc=3, ndf=64, n_layers_D=3).cuda().train()
    # net = nn.DataParallel(net)

    params = list(netD.parameters())
    optimizer_D = torch.optim.Adam(params, lr=cfgs['lr_D'], betas=(0.95, 0.999))
    
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, netD, optimizer, optimizer_D)


def train(net, netD, optimizer, optimizer_D):
    curr_iter = cfgs['last_iter']

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_t_record, loss_a_record = AvgMeter(), AvgMeter()
        loss_D_record = AvgMeter()
        loss_GAN_record = AvgMeter()

        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']

            haze, gt_trans_map, gt_ato, gt, _ = data

            batch_size = haze.size(0)

            haze = haze.cuda()
            gt_trans_map = gt_trans_map.cuda()
            gt_ato = gt_ato.cuda()
            gt = gt.cuda()

            # optimizer.zero_grad()

            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4, t, a = net(haze)
            
            criterionGAN = GANLoss(device=haze.device)
            
            fake_pool = ImagePool(pool_size=0)
            # fake_query = fake_pool.query(torch.cat((haze, x_jf.detach()), dim=1))
            fake_query = fake_pool.query(x_jf.detach())
            pred_fake_pool = netD.forward(fake_query)
            loss_D_fake = criterionGAN(pred_fake_pool, False)
            
            # fake_query_0 = fake_pool.query(torch.cat((haze, x_j0.detach()), dim=1))
            fake_query_0 = fake_pool.query(x_j0.detach())
            pred_fake_pool_0 = netD.forward(fake_query_0)
            loss_D_fake_0 = criterionGAN(pred_fake_pool_0, False)
            
            # fake_query_1 = fake_pool.query(torch.cat((haze, x_j1.detach()), dim=1))
            fake_query_1 = fake_pool.query(x_j1.detach())
            pred_fake_pool_1 = netD.forward(fake_query_1)
            loss_D_fake_1 = criterionGAN(pred_fake_pool_1, False)
            
            # fake_query_2 = fake_pool.query(torch.cat((haze, x_j2.detach()), dim=1))
            fake_query_2 = fake_pool.query(x_j2.detach())
            pred_fake_pool_2 = netD.forward(fake_query_2)
            loss_D_fake_2 = criterionGAN(pred_fake_pool_2, False)
            
            # fake_query_3 = fake_pool.query(torch.cat((haze, x_j3.detach()), dim=1))
            fake_query_3 = fake_pool.query(x_j3.detach())
            pred_fake_pool_3 = netD.forward(fake_query_3)
            loss_D_fake_3 = criterionGAN(pred_fake_pool_3, False)
            
            # fake_query_4 = fake_pool.query(torch.cat((haze, x_j4.detach()), dim=1))
            fake_query_4 = fake_pool.query(x_j4.detach())
            pred_fake_pool_4 = netD.forward(fake_query_4)
            loss_D_fake_4 = criterionGAN(pred_fake_pool_4, False)
            
            # pred_real = netD.forward(torch.cat((haze, gt.detach()), dim=1))
            pred_real = netD.forward(gt.detach())
            loss_D_real = criterionGAN(pred_real, True)
            
            loss_D = (loss_D_fake + loss_D_fake_0 + loss_D_fake_1 \
                    + loss_D_fake_2 + loss_D_fake_3 + loss_D_fake_4 \
                        + loss_D_real * 6) / 12
            
            
            
            # torch.autograd.set_detect_anomaly(True)
            
            

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            
            loss_x_jf = criterionVGG(x_jf, gt)
            loss_x_j0 = criterionVGG(x_j0, gt)
            loss_x_j1 = criterionVGG(x_j1, gt)
            loss_x_j2 = criterionVGG(x_j2, gt)
            loss_x_j3 = criterionVGG(x_j3, gt)
            loss_x_j4 = criterionVGG(x_j4, gt)

            loss_t = criterion(t, gt_trans_map)
            loss_a = criterion(a, gt_ato)

            loss = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4 \
                   + 10 * loss_t + loss_a
                   
            for p in netD.parameters():
                p.requires_grad=False
            
            # GAN loss (Fake Passability Loss)  
            loss_G_GAN = 0. 
            # pred_fake = netD.forward(torch.cat((haze, x_jf), dim=1))
            # pred_fake_0 = netD.forward(torch.cat((haze, x_j0), dim=1))
            # pred_fake_1 = netD.forward(torch.cat((haze, x_j1), dim=1))
            # pred_fake_2 = netD.forward(torch.cat((haze, x_j2), dim=1))
            # pred_fake_3 = netD.forward(torch.cat((haze, x_j3), dim=1))
            # pred_fake_4 = netD.forward(torch.cat((haze, x_j4), dim=1))
            pred_fake = netD(x_jf)
            pred_fake_0 = netD(x_j0)
            pred_fake_1 = netD(x_j1)
            pred_fake_2 = netD(x_j2)
            pred_fake_3 = netD(x_j3)
            pred_fake_4 = netD(x_j4)
            # pred_fake_0 = netD(x_j0)
            loss_G_GAN += criterionGAN(pred_fake, True)
            loss_G_GAN += criterionGAN(pred_fake_0, True)
            loss_G_GAN += criterionGAN(pred_fake_1, True)
            loss_G_GAN += criterionGAN(pred_fake_2, True)
            loss_G_GAN += criterionGAN(pred_fake_3, True)
            loss_G_GAN += criterionGAN(pred_fake_4, True)
            loss_G_GAN /= 6
            
            # num_D = 1
            loss_G_GAN_Feat = 0
            # n_layers_D = 3
            # lambda_feat = 10.0
            # feat_weights = 4.0 / (n_layers_D + 1)
            # D_weights = 1.0 / num_D
            # for i in range(num_D):
            #     for j in range(len(pred_fake[i])-1):
            #         loss_G_GAN_Feat += D_weights * feat_weights * \
            #             criterion(pred_fake[i][j], pred_real[i][j].detach()) * lambda_feat
            # loss_G_GAN_Feat += cal_FM(criterion=criterion, pred_fake=pred_fake, pred_real=pred_real)
            # loss_G_GAN_Feat += cal_FM(criterion=criterion, pred_fake=pred_fake_0, pred_real=pred_real)
            # loss_G_GAN_Feat += cal_FM(criterion=criterion, pred_fake=pred_fake_1, pred_real=pred_real)
            # loss_G_GAN_Feat += cal_FM(criterion=criterion, pred_fake=pred_fake_2, pred_real=pred_real)
            # loss_G_GAN_Feat += cal_FM(criterion=criterion, pred_fake=pred_fake_3, pred_real=pred_real)
            # loss_G_GAN_Feat += cal_FM(criterion=criterion, pred_fake=pred_fake_4, pred_real=pred_real)
            # loss_G_GAN_Feat /= 6
            
            # loss_G = loss_G_GAN + loss_G_GAN_Feat
            loss_G = loss_G_GAN
            
            optimizer.zero_grad() 
            # print(loss_G_GAN)
            loss_G.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            
            for p in netD.parameters():
                p.requires_grad=True

            # update recorder
            train_loss_record.update(loss.item(), batch_size)
            
            loss_D_record.update(loss_D.item(), batch_size)
            loss_GAN_record.update(loss_G.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j0_record.update(loss_x_j0.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)

            loss_t_record.update(loss_t.item(), batch_size)
            loss_a_record.update(loss_a.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_D %.5f], [loss_GAN %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
                  '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f], [loss_t %.5f], [loss_a %.5f], ' \
                  '[lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss_D_record.avg, loss_GAN_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
                   loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                   loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, netD, curr_iter, optimizer, optimizer_D)

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, netD, curr_iter, optimizer, optimizer_D):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'])
    print('[validate]: [iter %d], [loss %.5f]' % (curr_iter + 1, loss_record.avg))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))
    torch.save(netD.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_D.pth'))
    torch.save(optimizer_D.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_D_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    print(torch.cuda.device_count())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=8)

    criterion = nn.L1Loss().cuda()
    criterionVGG = VGGLoss(gpu_ids=args.gpus).cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
