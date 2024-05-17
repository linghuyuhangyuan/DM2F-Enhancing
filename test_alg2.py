# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, TEST_HAZERD_ROOT, TEST_MY_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, HazerdDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.functional.image import structural_similarity_index_measure
from metric_cal import ciede2000, mse
from networks import define_D, GANLoss

print(torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# torch.manual_seed(2018)
torch.cuda.set_device(0)
device = "cuda:1"

ckpt_path = 'ckpts/ckpt_alg2'
exp_name = 'RESIDE_ITS'
# exp_name = 'O-Haze'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_20000_loss_0.05082_lr_0.000000',
    'snapshot': 'iter_3000_loss_0.05215_lr_0.000432',
    # 'snapshot': 'resnext_101_32x4d',
}

to_test = {
    'SOTS': TEST_HAZERD_ROOT,
    # 'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().to(device)
        criterionGAN = GANLoss(device=device).to(device)

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet(use_refine=False, use_sep=False).to(device)
                dataset = HazerdDataset(root)
                # dataset = OHazeDataset(root, 'test')
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().to(device)
                dataset = OHazeDataset(root, 'test')
            else:
                raise NotImplementedError
            
            netD = define_D(input_nc=3, ndf=64, n_layers_D=3).to(device)

            # net = nn.DataParallel(net)
            # ngpu = torch.cuda.device_count()
            # if (ngpu > 1):
            #     net = nn.DataParallel(net, list(range(ngpu)))

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
                netD.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_D.pth')))
                # state_dict = torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'))
                # # 去除module.前缀
                # if 'module.' in list(state_dict.keys())[0]:
                #     state_dict = {k[7:]: v for k, v in state_dict.items()}
                # net.load_state_dict(state_dict)
                
            net.eval()
            netD.eval()
            
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, delta_es, mse_values = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.to(device)
                haze_origin = haze.clone()
                
                loss_D = 9999
                num = 0
                epsilon = 0.2
                num_limit = 0
                while loss_D > epsilon:
                    if 'O-Haze' in name:
                        haze = sliding_forward(net, haze).detach()
                    else:
                        haze = net(haze).detach()
                    
                    
                    pred = netD.forward(haze)
                    loss_D = criterionGAN(pred, True)
                    print(loss_D)
                    
                    if num >= num_limit:
                        break
                    else:
                        num += 1
                    
                    
                res = haze

                loss = criterion(res, gts.to(device))
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    # print(gt)
                    # print(r.shape)
                    h = r.shape[0]
                    w = r.shape[1]
                    c = r.shape[2]
                    # r = r.reshape((1, 3, h, w))
                    # gt = gt.reshape((1, 3, h, w))
                    r_tensor = torch.tensor(r)
                    gt_tensor = torch.tensor(gt)
                    # print(r.shape)
                    ssim = structural_similarity(gt, r, data_range=1, channel_axis=-1,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    # ssim = structural_similarity(gt, r, data_range=1, multichannel=True, win_size=3,
                    #                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    # ssim = structural_similarity_index_measure(gt_tensor, r_tensor)
                    ssims.append(ssim)
                    mse_value = mse(gt, r)
                    mse_values.append(mse_value)
                    # print(mse_value)
                    # print(r_tensor.shape)
                    # print(r_tensor)
                    delta_e = np.mean(ciede2000(rgbsample=r, rgbstd=gt))
                    # print(delta_e)
                    delta_es.append(delta_e)
                    # print(torch.mean(delta_e))
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}, MSE {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, delta_e, mse_value))


                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(delta_es):.6f}, MSE: {np.mean(mse_values):.6f}")


if __name__ == '__main__':
    main()
