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
from networks import VGGLoss, LocalEnhancer, weights_init
from unet import UNet,RefinementNet, MultiColor_Fusion, MultiColor_direct_fusion
from torch.backends import cudnn

print(torch.cuda.device_count())
# gpus = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
# torch.cuda.set_device(1)

torch.manual_seed(2018)
torch.cuda.set_device(0)
device = "cuda:0"

ckpt_path = 'ckpts/ckpt'
ckpt_path_2 = "ckpts/ckpt_alg1"
# exp_name = 'RESIDE_ITS'
exp_name = 'O-Haze'
exp_name_2 = 'RESIDE_ITS'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_20000_loss_0.05082_lr_0.000000',
    'snapshot': 'iter_20000_loss_0.04937_lr_0.000000',
    'snapshot_2': 'iter_22000_loss_0.97667_lr_0.000244',
    # 'snapshot': 'resnext_101_32x4d',
}

to_test = {
    # 'SOTS': TEST_HAZERD_ROOT,
    'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                # net_refine = LocalEnhancer(input_nc=3, output_nc=3).cuda()
                # net_refine = UNet(in_channels=3, out_channels=3).cuda()
                net_refine = RefinementNet().cuda()
                dataset = HazerdDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                # net_refine = LocalEnhancer(input_nc=3, output_nc=3).cuda()
                # net_refine = UNet(in_channels=3, out_channels=3).cuda()
                # net_refine = MultiColor_direct_fusion().cuda()
                net_refine = MultiColor_Fusion(training=False).cuda()
                dataset = OHazeDataset(root, 'test')
                
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)
            # ngpu = torch.cuda.device_count()
            # if (ngpu > 1):
            #     net = nn.DataParallel(net, list(range(ngpu)))

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                # print(net.device)
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
                # print(os.path.join(ckpt_path_2, exp_name, args['snapshot_2'] + '.pth'))
                refine_path = os.path.join(ckpt_path_2, exp_name_2, args['snapshot_2'] + '.pth')
                # print(refine_path)
                net_refine.load_state_dict(torch.load(refine_path))
                # state_dict = torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'))
                # # 去除module.前缀
                # if 'module.' in list(state_dict.keys())[0]:
                #     state_dict = {k[7:]: v for k, v in state_dict.items()}
                # net.load_state_dict(state_dict)
                # exit(0)
                
            net.eval()
            net_refine.eval()
            
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, delta_es, mse_values = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path_2, exp_name_2,
                                         '(%s) %s_%s' % (exp_name_2, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    # print(net.device())
                    # print(haze.device())
                    res = sliding_forward(net, haze, device=device).detach()
                    # res = net_refine(res, haze).detach()
                    res = res + sliding_forward(net_refine, x=res, x2=haze, special=True, device=device).detach()
                else:
                    res = net_refine(net(haze), haze).detach()

                loss = criterion(res, gts.cuda())
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
                        os.path.join(ckpt_path_2, exp_name_2,
                                     '(%s) %s_%s' % (exp_name_2, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(delta_es):.6f}, MSE: {np.mean(mse_values):.6f}")


if __name__ == '__main__':
    main()
