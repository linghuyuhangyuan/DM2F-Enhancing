import numpy as np
import cv2

def mse(rgb_1, rgb_2):
    # 确保两个图像的形状相同
    assert rgb_1.shape == rgb_2.shape, "输入图像形状不匹配"
    
    # 计算每个像素通道之间的差异
    diff = rgb_1 - rgb_2
    
    # 将差异平方并求和
    squared_diff = np.square(diff)
    sum_squared_diff = np.sum(squared_diff)
    
    # 将结果除以像素数得到 MSE
    mse_value = sum_squared_diff / (rgb_1.shape[0] * rgb_1.shape[1])
    
    return mse_value


def ciede2000(rgbstd, rgbsample, KLCH=None):
    """
    Copy from matlab code: HazeRD and edit it.
    """
    labstd = cv2.cvtColor(rgbstd, cv2.COLOR_BGR2Lab)
    labsample = cv2.cvtColor(rgbsample, cv2.COLOR_BGR2Lab)
    
    h = labstd.shape[0]
    w = labstd.shape[1]
    c = labstd.shape[2]
    
    Labstd = labstd.reshape(h*w, c)
    Labsample = labsample.reshape(h*w, c)

    # Ensure inputs are numpy arrays
    Labstd = np.array(Labstd)
    Labsample = np.array(Labsample)
    
    # Check input dimensions
    if Labstd.shape != Labsample.shape:
        print('deltaE00: Standard and Sample sizes do not match')
        return None
    if Labstd.shape[1] != 3:
        print('deltaE00: Standard and Sample Lab vectors should be Kx3 vectors')
        return None
    
    # Set default parametric factors if not provided
    if KLCH is None:
        kl, kc, kh = 1, 1, 1
    else:
        if KLCH.shape != (3,):
            print('deltaE00: KLCH must be a 1x3 vector')
            return None
        kl, kc, kh = KLCH
    
    Lstd, astd, bstd = Labstd[:, 0], Labstd[:, 1], Labstd[:, 2]
    Lsample, asample, bsample = Labsample[:, 0], Labsample[:, 1], Labsample[:, 2]
    
    Cabstd = np.sqrt(astd**2 + bstd**2)
    Cabsample = np.sqrt(asample**2 + bsample**2)
    Cabarithmean = (Cabstd + Cabsample) / 2
    
    G = 0.5 * (1 - np.sqrt((Cabarithmean**7) / (Cabarithmean**7 + 25**7)))
    
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    Cpstd = np.sqrt(apstd**2 + bstd**2)
    Cpsample = np.sqrt(apsample**2 + bsample**2)
    
    Cpprod = Cpsample * Cpstd
    zcidx = np.where(Cpprod == 0)[0]
    
    hpstd = np.arctan2(bstd, apstd)
    hpstd = np.mod(hpstd + 2 * np.pi * (hpstd < 0), 2 * np.pi)
    hpstd[(np.abs(apstd) + np.abs(bstd)) == 0] = 0
    
    hpsample = np.arctan2(bsample, apsample)
    hpsample = np.mod(hpsample + 2 * np.pi * (hpsample < 0), 2 * np.pi)
    hpsample[(np.abs(apsample) + np.abs(bsample)) == 0] = 0
    
    dL = Lsample - Lstd
    dC = Cpsample - Cpstd
    
    dhp = hpsample - hpstd
    dhp[dhp > np.pi] -= 2 * np.pi
    dhp[dhp < -np.pi] += 2 * np.pi
    dhp[zcidx] = 0
    
    ΔH_ = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    ΔH__bar = np.abs(hpstd - hpsample)
    ΔH__bar[np.abs(hpstd - hpsample) > np.pi] -= 2 * np.pi
    ΔH__bar = np.abs(ΔH__bar)
    
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    
    hp = (hpstd + hpsample) / 2
    hp[ΔH__bar > np.pi] -= np.pi
    
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + 0.32 * np.cos(3 * hp + np.pi / 30) - 0.2 * np.cos(4 * hp - 63 * np.pi / 180)
    Sl = 1 + 0.015 * (Lp - 50)**2 / np.sqrt(20 + (Lp - 50)**2)
    Sc = 1 + 0.045 * Cp
    Sh = 1 + 0.015 * Cp * T
    
    delthetarad = (30 * np.pi / 180) * np.exp(-((180 / np.pi * hp - 275) / 25)**2)
    Rc = 2 * np.sqrt((Cp**7) / (Cp**7 + 25**7))
    RT = -np.sin(2 * delthetarad) * Rc
    
    dE00 = np.sqrt((dL / (kl * Sl))**2 + (dC / (kc * Sc))**2 + (ΔH_ / (kh * Sh))**2 + RT * (dC / (kc * Sc)) * (ΔH_ / (kh * Sh)))
    
    return dE00


# import torch
# import torch.nn.functional as F
# import math
# import cv2
# import numpy as np

# def ciede2000(rgb1, rgb2):
#     lab1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2Lab)
#     lab2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2Lab)
    
#     # 提取 Lab 颜色通道
#     L1, a1, b1 = lab1[:,:,0], lab1[:,:,1], lab1[:,:,2]
#     L2, a2, b2 = lab2[:,:,0], lab2[:,:,1], lab2[:,:,2]
    
#     # 计算色彩信息
#     C1 = np.sqrt(a1 ** 2 + b1 ** 2)
#     C2 = np.sqrt(a2 ** 2 + b2 ** 2)
#     C_ave = (C1 + C2) / 2
#     G = 0.5 * (1 - np.sqrt(C_ave ** 7 / (C_ave ** 7 + 6103515625)))  # 6103515625 = 25 ** 7
    
#     a1_ = (1 + G) * a1
#     a2_ = (1 + G) * a2
    
#     C1_ = np.sqrt(a1_ ** 2 + b1 ** 2)
#     C2_ = np.sqrt(a2_ ** 2 + b2 ** 2)
    
#     h1_ = np.arctan2(b1, a1_)
#     h2_ = np.arctan2(b2, a2_)
#     h1_[h1_ < 0] += 2 * np.pi
#     h2_[h2_ < 0] += 2 * np.pi
    
#     dL_ = L2 - L1
#     dC_ = C2_ - C1_
#     dH_ = 2 * np.sqrt(C1_ * C2_) * np.sin((h2_ - h1_) / 2)
#     dH_[np.isnan(dH_)] = 0  # 处理除零错误
    
#     C_ave = (C1_ + C2_) / 2
#     h_ave = (h1_ + h2_) / 2
#     h_ave[h_ave > np.pi] -= 2 * np.pi
    
#     T = 1 - 0.17 * np.cos(h_ave - np.pi / 6) + 0.24 * np.cos(2 * h_ave) + 0.32 * np.cos(3 * h_ave + np.pi / 30) - 0.2 * np.cos(4 * h_ave - 63 * np.pi / 180)
    
#     dTheta = 30 * np.exp(-((h_ave * 180 / np.pi - 275) / 25) ** 2)
    
#     R_C = 2 * np.sqrt(C_ave ** 7 / (C_ave ** 7 + 6103515625))
#     S_C = 1 + 0.045 * C_ave
#     S_H = 1 + 0.015 * C_ave * T
    
#     dL_ /= 1
#     dC_ /= S_C
#     dH_ /= S_H
    
#     dE_00 = np.sqrt(dL_ ** 2 + dC_ ** 2 + dH_ ** 2 + R_C * dC_ * dH_)
    
#     # 计算平均值
#     dE_00_mean = np.mean(dE_00)
    
#     return dE_00_mean
