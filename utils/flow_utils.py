#@代码实现了将光流转换为颜色编码图像（flow_to_image 函数）、
# 计算光流颜色图（compute_color 函数），还包括一些其他与光流相关的操作，
# 如重采样、变形和一致性检查。
import os
import cv2
import numpy as np
from PIL import Image
from os.path import *
UNKNOWN_FLOW_THRESH = 1e7 #定义了光流中无效值的阈值，超过这个值的光流被认为是无效的


def flow_to_image(flow, global_max=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    将光流转换为颜色编码的图像
    :param flow: 光流数据（二维数组）
    :return: 使用Middlebury颜色编码的光流图像
    """
    u = flow[:, :, 0]

    v = flow[:, :, 1]
    """代表光流在水平方向和垂直方向的位移"""
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    """定义最大和最小值"""
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    """
    筛选出无效的光流值（即大于阈值的值），这些值会被设为 0
    UNKNOWN_FLOW_THRESH = 1e7
    """
    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    """计算每个像素处的光流幅度"""
    if global_max == None:
        maxrad = max(-1, np.max(rad))
    else:
        maxrad = global_max
    """如果没有给定 global_max，则使用最大幅度 maxrad 来归一化 u 和 v"""
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)
    """函数根据归一化的 u 和 v, 调用了 compute_color 函数来生成颜色编码的光流图像生成颜色编码的光流图像"""
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    '''对无效的光流区域（即 idxUnknow 为 True 的区域），将颜色图像中的对应像素设置为 0（黑色），这样这些区域就不会被显示。'''
    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    根据 u 和 v 计算光流的颜色图像
    :param u: 水平方向的光流
    :param v: 垂直方向的光流
    :return: 颜色编码的光流图像
    """
    [h, w] = u.shape
    '''获取图像尺寸'''
    img = np.zeros([h, w, 3])
    '''初始化一个三通道（RGB）的图像'''
    nanIdx = np.isnan(u) | np.isnan(v)
    '''标记无效区域，并设为0'''
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    '''
    调用 make_color_wheel() 函数生成颜色轮，
    这是一个二维数组，每一行表示一种颜色。颜色轮用于将光流的方向映射到颜色
    '''
    ncols = np.size(colorwheel, 0)
    '''颜色论的颜色数'''
    rad = np.sqrt(u**2+v**2)
    '''光流的幅度'''
    a = np.arctan2(-v, -u) / np.pi
    '''光流的角度，用于决定颜色
    计算从负 x 轴逆时针旋转的角度。结果除以 π，使得角度的取值范围为 [-1, 1]
    '''
    fk = (a+1) / 2 * (ncols - 1) + 1
    '''将光流角度 a 转换为在颜色轮中对应的颜色索引。
    将角度范围 [-1, 1] 映射到颜色轮的索引范围 [1, ncols]'''
    k0 = np.floor(fk).astype(int) #计算颜色轮中光流角度对应的下界索引

    k1 = k0 + 1 #计算颜色轮中光流角度对应的上界索引
    k1[k1 == ncols+1] = 1 #如果上界超出了颜色轮的最大值，则将其设置为 1
    f = fk - k0 #计算插值因子，用于在颜色轮中进行线性插值，生成平滑的颜色过渡

    for i in range(0, np.size(colorwheel,1)): #对于每个颜色通道，从 0 到颜色轮的通道数，颜色论的每一行是一个颜色，3代表RBG通道
        tmp = colorwheel[:, i] #取出颜色轮的第i个通道的颜色值，这个数组的长度是颜色轮的总颜色数（ncols），其中每个值表示该通道对应的颜色
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1  #按照索引k来获取光流角度的颜色分量，f插值因子过渡两个颜色。

        idx = rad <= 1 #又是一个布尔掩码，标记光流幅度小于等于 1 的位置
        #对于光流幅度较小的区域，将其颜色亮度做非线性调整：
        # 当幅度rad接近0时，col[idx]的值趋向于1（白色），这是为了保持较小光流的区域亮度接近白色。
        # 当幅度rad接近1时，col[idx]保持原始颜色。这种处理使得小幅度的光流颜色更加显著，防止颜色变得过暗
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx) #获取光流幅度大于 1 的区域（即 idx 为 False 的位置）

        col[notidx] *= 0.75 #将幅度大于 1 的区域的颜色亮度乘以 0.75，进行暗化处理。这样可以抑制高幅度光流区域的颜色过于亮，保持图像的整体对比度。
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx))) #更新图像的颜色通道
    '''
    使用颜色轮对光流的角度和幅度进行颜色编码。
    通过插值法计算颜色（col0 和 col1），
    生成平滑的颜色过渡。'''
    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    根据 Middlebury 颜色编码生成颜色轮
    :return: 颜色轮
    """
    RY = 15  # 红到黄的过渡段长度 (Red to Yellow)
    YG = 6  # 黄到绿的过渡段长度 (Yellow to Green)
    GC = 4  # 绿到青的过渡段长度 (Green to Cyan)
    CB = 11  # 青到蓝的过渡段长度 (Cyan to Blue)
    BM = 13  # 蓝到洋红的过渡段长度 (Blue to Magenta)
    MR = 6  # 洋红到红的过渡段长度 (Magenta to Red)

    ncols = RY + YG + GC + CB + BM + MR
    #计算颜色轮的总颜色数量
    colorwheel = np.zeros([ncols, 3])
    #创建颜色轮矩阵
    col = 0 #定义颜色轮的起始索引。用于跟踪当前颜色段在颜色轮中的起始位置

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def resize_flow(flow, H_new, W_new): #调整光流图的尺寸，同时根据新旧图像尺寸的比例来正确调整光流向量
    H_old, W_old = flow.shape[0:2]# 获取原始光流图的高度和宽度
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    '''使用 OpenCV 的 resize 函数将光流图的尺寸调整到 (W_new, H_new)。
    INTER_LINEAR 是一种双线性插值方法，适用于调整图像大小。'''
    flow_resized[:, :, 0] *= H_new / H_old  # 按比例调整光流在x方向（水平）的分量
    flow_resized[:, :, 1] *= W_new / W_old  # 按比例调整光流在y方向（垂直）的分量
    return flow_resized



def warp_flow(img, flow): #根据光流信息对图像进行扭曲变换
    h, w = flow.shape[:2]
    flow_new = flow.copy() #生成光流的副本 flow_new，以避免修改原始光流
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis] #光流分量于坐标值相加，指示出改目标坐标

    res = cv2.remap(img, flow_new, None, #OpenCV 的 remap 函数根据 flow_new 中的位置信息重新映射图像 img。它将原始图像中的每个像素移动到新的坐标位置，完成图像的扭曲变换。
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT)
    '''
                    INTER_CUBIC：使用三次插值法，这种方法能在图像变换时提供较高的图像质量。
                    borderMode=cv2.BORDER_CONSTANT：定义边界模式，处理那些在扭曲后超出边界的像素值（设置为常量填充）。
                '''
    return res


def consistCheck(flowB, flowF):  #用于检查光流的一致性，通过前后光流的比较来检测光流的偏差

    # |--------------------|  |--------------------|
    # |       y            |  |       v            |
    # |   x   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    # sub: numPix * [y x t]

    imgH, imgW, _ = flowF.shape

    (fy, fx) = np.mgrid[0 : imgH, 0 : imgW].astype(np.float32)
    fxx = fx + flowB[:, :, 0]  # horizontal水平
    fyy = fy + flowB[:, :, 1]  # vertical垂直，根据反向光流 flowB 计算变形后的像素坐标

    u = (fxx + cv2.remap(flowF[:, :, 0], fxx, fyy, cv2.INTER_LINEAR) - fx)
    v = (fyy + cv2.remap(flowF[:, :, 1], fxx, fyy, cv2.INTER_LINEAR) - fy)
    #将正向光流 flowF 映射到反向光流变形后的坐标 (fxx, fyy)。u 和 v 分别代表 x 和 y 方向上的光流偏差。
    BFdiff = (u ** 2 + v ** 2) ** 0.5
    #计算前后光流偏差的欧几里得距离（L2 范数），用于衡量光流的一致性。

    return BFdiff, np.stack((u, v), axis=2)


def read_optical_flow(basedir, img_i_name, read_fwd): #读取光流文件
    flow_dir = os.path.join(basedir, 'flow')

    fwd_flow_path = os.path.join(flow_dir, '%s_fwd.npz'%img_i_name[:-4])
    bwd_flow_path = os.path.join(flow_dir, '%s_bwd.npz'%img_i_name[:-4])

    if read_fwd:
      fwd_data = np.load(fwd_flow_path) #np.load 从 .npz 文件中加载光流数据。
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      return fwd_flow, fwd_mask
    else:
      bwd_data = np.load(bwd_flow_path)
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      return bwd_flow, bwd_mask


def compute_epipolar_distance(T_21, K, p_1, p_2): #计算两个点之间的极线几何距离
    '''T_21：从第一个视角到第二个视角的变换矩阵（包括旋转和平移）。
    K：相机的内参矩阵。
    p_1, p_2：分别表示两帧中的像素点。'''
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    '''利用平移向量的反对称矩阵（skew(t_21)）与旋转矩阵相乘，得到本质矩阵 E_mat'''
    # compute bearing vector
    inv_K = np.linalg.inv(K)
    '''计算相机内参矩阵的逆。'''
    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)
    '''计算基础矩阵 F_mat，这是通过内参矩阵和本质矩阵得到的。基础矩阵用于描述对应点之间的几何关系。'''
    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :]**2 + l_2[1, :]**2) + 1e-8
    geometric_e_distance = algebric_e_distance/n_term #计算代数极线距离，即点 p_2 到极线 l_2 的投影距离。
    geometric_e_distance = np.abs(geometric_e_distance) #计算几何极线距离，归一化的代数距离，用于衡量点与极线的几何关系。

    return geometric_e_distance


def skew(x):  #反对称矩阵
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
