import os
import re
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import matplotlib
import torch
import lpips
import pywt
from scipy.signal import butter, filtfilt, medfilt, welch
from pyVHR.analysis import Pipeline
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- 系统配置 ---
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
matplotlib.use('Agg')

# ==================== 【0. 实验对象批量配置区】 ====================
SUBJECT_VIDEO_LIST = [
    # "/tank/数据集/UBFC/UBFC2/subject27/vid.avi",
    "/tank/数据集/UBFC/UBFC2/subject31/vid.avi",
    # "/tank/数据集/UBFC/UBFC2/subject15/vid.avi",
    # "/tank/数据集/UBFC/UBFC2/subject38/vid.avi"
    # 在这里继续添加你需要测试的其他 subject...
]

# 统一输出总目录
OUT_BASE_DIR = "/tank/在读/zhaobowen/qim_data/central_unified_comprehensive_comparison_final/"


# ==================== 【1. 信号后处理与评估工具】 ====================
def filter_bvp_signal(signal, fps):
    smoothed_signal = medfilt(signal, kernel_size=5)
    low_freq, high_freq = 0.7, 2.5
    nyquist_freq = 0.5 * fps
    b, a = butter(3, [low_freq / nyquist_freq, high_freq / nyquist_freq], btype='band')
    return filtfilt(b, a, smoothed_signal)


class SignalMetricsEvaluator:
    @staticmethod
    def _get_sliding_windows(signal, fps, window_sec=12, step_sec=1):
        window_size = int(fps * window_sec)
        step_size = int(fps * step_sec)
        windows = []
        if len(signal) < window_size: return [signal]
        for start in range(0, len(signal) - window_size + 1, step_size):
            windows.append(signal[start:start + window_size])
        return windows

    @staticmethod
    def _calculate_window_bpm_snr(window, fps):
        f, pxx = welch(window, fs=fps, nperseg=len(window))
        valid_idx = np.where((f >= 0.5) & (f <= 4.0))[0]
        if len(valid_idx) == 0: return 0.0, 0.0
        f_valid = f[valid_idx]
        pxx_valid = pxx[valid_idx]
        peak_idx = np.argmax(pxx_valid)
        f_hr = f_valid[peak_idx]
        bpm = f_hr * 60.0
        window_hz = 0.1
        signal_mask = np.zeros_like(pxx_valid, dtype=bool)
        signal_mask[(f_valid >= f_hr - window_hz) & (f_valid <= f_hr + window_hz)] = True
        signal_power = np.sum(pxx_valid[signal_mask])
        noise_power = np.sum(pxx_valid[~signal_mask])
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        return bpm, snr

    @classmethod
    def evaluate_metrics(cls, orig_signal, rec_signal, fps, window_sec=12, step_sec=1):
        orig_windows = cls._get_sliding_windows(orig_signal, fps, window_sec, step_sec)
        rec_windows = cls._get_sliding_windows(rec_signal, fps, window_sec, step_sec)
        min_len = min(len(orig_windows), len(rec_windows))
        bpm_maes, rec_snrs = [], []
        for i in range(min_len):
            orig_bpm, _ = cls._calculate_window_bpm_snr(orig_windows[i], fps)
            rec_bpm, rec_snr = cls._calculate_window_bpm_snr(rec_windows[i], fps)
            if orig_bpm > 0 and rec_bpm > 0:
                bpm_maes.append(abs(orig_bpm - rec_bpm))
                rec_snrs.append(rec_snr)
        return np.mean(bpm_maes) if bpm_maes else float('nan'), np.mean(rec_snrs) if rec_snrs else float('nan')


class VisualMetricsEvaluator:
    _lpips_model = None

    @classmethod
    def evaluate_video_quality(cls, orig_video_path, wm_video_path, sample_interval=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cls._lpips_model is None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls._lpips_model = lpips.LPIPS(net='alex').to(device)

        cap_orig = cv2.VideoCapture(orig_video_path)
        cap_wm = cv2.VideoCapture(wm_video_path)
        psnr_list, ssim_list, lpips_list = [], [], []
        frame_idx = 0
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_wm, frame_wm = cap_wm.read()
            if not ret_orig or not ret_wm: break
            if frame_idx % sample_interval == 0:
                p_val = psnr(frame_orig, frame_wm, data_range=255)
                psnr_list.append(p_val)
                gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                gray_wm = cv2.cvtColor(frame_wm, cv2.COLOR_BGR2GRAY)
                ssim_list.append(ssim(gray_orig, gray_wm, data_range=255))
                img0 = lpips.im2tensor(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)).to(device)
                img1 = lpips.im2tensor(cv2.cvtColor(frame_wm, cv2.COLOR_BGR2RGB)).to(device)
                with torch.no_grad(): lpips_list.append(cls._lpips_model(img0, img1).item())
            frame_idx += 1
        cap_orig.release()
        cap_wm.release()
        return np.mean(psnr_list) if psnr_list else 0, np.mean(ssim_list) if ssim_list else 0, np.mean(
            lpips_list) if lpips_list else 0


def compress_video_ffmpeg(input_path, output_path, codec='libx264', qp_val=23):
    cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', codec, '-qp', str(qp_val), '-preset', 'fast', '-pix_fmt',
           'yuv420p', output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


class HammingECC:
    @staticmethod
    def encode_16to28(val_16bit):
        bits = format(val_16bit, '016b')
        encoded_bits = []
        for i in range(0, 16, 4):
            nibble = [int(x) for x in bits[i:i + 4]]
            d1, d2, d3, d4 = nibble
            p1 = d1 ^ d2 ^ d4;
            p2 = d1 ^ d3 ^ d4;
            p3 = d2 ^ d3 ^ d4
            encoded_bits.extend([p1, p2, d1, p3, d2, d3, d4])
        return encoded_bits

    @staticmethod
    def decode_28to16(bits_28):
        decoded_bits = []
        for i in range(0, 28, 7):
            block = bits_28[i:i + 7]
            p1, p2, d1, p3, d2, d3, d4 = block
            s1 = p1 ^ d1 ^ d2 ^ d4;
            s2 = p2 ^ d1 ^ d3 ^ d4;
            s3 = p3 ^ d2 ^ d3 ^ d4
            syndrome = s1 * 1 + s2 * 2 + s3 * 4
            error_pos = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
            if syndrome != 0 and syndrome in error_pos: block[error_pos[syndrome]] ^= 1
            decoded_bits.extend([block[2], block[4], block[5], block[6]])
        return int("".join(map(str, decoded_bits)), 2)


# ==================== 【2. 算法矩阵库】 ====================

class BaseBlindAdaptiveWatermarker:
    """提取公共逻辑：固定底部阵地获取 (绝不截断，保留海量冗余投票)"""

    def __init__(self, block_size=16, payload_bits=28):
        self.block_size = block_size
        self.payload_bits = payload_bits

    # def _get_fixed_lower_coords(self, blocks_h, blocks_w):
    #     coords = []
    #     # 选取底部大约 1/3 的区域，完美避开上半部人脸
    #     start_i = int(blocks_h * 0.66)
    #     for i in range(start_i, blocks_h):
    #         for j in range(blocks_w):
    #             coords.append((i, j))
    #     # 返回上千个坐标，利用 idx % 28 自动实现多数投票冗余
    #     return coords

    def _get_fixed_lower_coords(self, blocks_h, blocks_w):
        """
        【精准打击版】：只选取受试者的躯干/衣服区域，大幅缩小视觉影响范围。
        """
        coords = []

        # 1. 垂直方向：只取画面最底部的 25% ~ 30% (完美避开下巴和脖子)
        start_i = int(blocks_h * 0.75)

        # 2. 水平方向：切掉左右两侧各 20% 的纯色背景，只保留中间 60% 的受试者身体
        start_j = int(blocks_w * 0.20)
        end_j = int(blocks_w * 0.80)

        for i in range(start_i, blocks_h):
            for j in range(start_j, end_j):
                coords.append((i, j))

        # 返回躯干区域的坐标，供后续多数投票使用
        return coords

    def _get_tex_factor(self, block):
        """纹理自适应因子（仅 Proposed_JND 使用）"""
        var = np.var(block)
        return np.clip(var / 30.0, 0.5, 4.0)


# 2.1 你的方案：集中差分 DCT (保留自适应，画质碾压)
class JND_DifferentialWatermarker(BaseBlindAdaptiveWatermarker):
    def __init__(self, border_blocks=3):
        super().__init__()
        self.base_margin = 150
        self.coeff_pairs = [((1, 5), (5, 1)), ((2, 4), (4, 2)), ((1, 6), (6, 1)), ((3, 4), (4, 3)), ((2, 6), (6, 2)),
                            ((3, 5), (5, 3)), ((1, 8), (8, 1)), ((4, 5), (5, 4))]

    def embed_video(self, v_path, sig, out_path):
        cap = cv2.VideoCapture(v_path)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        b_min, b_max = sig.min(), sig.max()
        norm = np.clip(((sig - b_min) / (b_max - b_min)) * 65535, 0, 65535).astype(np.uint16)
        out = cv2.VideoWriter(out_path, 0, fps, (w, h))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or f_idx >= len(norm): break
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y = yuv[:, :, 0].astype(np.float32)
            bits = HammingECC.encode_16to28(norm[f_idx])
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                block = y[by:by + 16, bx:bx + 16]

                # 【特权】：仅本文方法享受纹理自适应强度保护画质
                margin = self.base_margin * self._get_tex_factor(block)

                dct = cv2.dct(block)
                bit = bits[idx % 28]
                As, Bs = sum(dct[y, x] for (y, x), _ in self.coeff_pairs), sum(
                    dct[y, x] for _, (y, x) in self.coeff_pairs)
                if bit == 1 and (As - Bs) < margin:
                    corr = (margin - (As - Bs)) / 16
                    for (y1, x1), (y2, x2) in self.coeff_pairs: dct[y1, x1] += corr; dct[y2, x2] -= corr
                elif bit == 0 and (Bs - As) < margin:
                    corr = (margin - (Bs - As)) / 16
                    for (y1, x1), (y2, x2) in self.coeff_pairs: dct[y1, x1] -= corr; dct[y2, x2] += corr
                y[by:by + 16, bx:bx + 16] = cv2.idct(dct)
            yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR));
            f_idx += 1
        cap.release();
        out.release()
        return True, (b_min, b_max), fps

    def extract_video(self, v_path):
        cap = cv2.VideoCapture(v_path)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        extr = []
        while True:
            ret, frame = cap.read();
            if not ret: break
            y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
            scores = np.zeros(28)
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                dct = cv2.dct(y[by:by + 16, bx:bx + 16])
                diff = sum(dct[y, x] for (y, x), _ in self.coeff_pairs) - sum(
                    dct[y, x] for _, (y, x) in self.coeff_pairs)
                if diff > 25:
                    scores[idx % 28] += 1
                elif diff < -25:
                    scores[idx % 28] -= 1
            extr.append(HammingECC.decode_28to16([1 if s > 0 else 0 for s in scores]))
        cap.release()
        return np.array(extr)


# 2.2 对照组：QIM DCT (恢复绝对固定强度)
class QIM_Watermarker(BaseBlindAdaptiveWatermarker):
    def __init__(self, border_blocks=3, delta=100):
        super().__init__()
        self.delta = delta
        self.coords = [(1, 5), (5, 1), (2, 4), (4, 2), (1, 6), (6, 1), (3, 4), (4, 3), (2, 6), (6, 2), (3, 5), (5, 3),
                       (1, 8), (8, 1), (4, 5), (5, 4)]

    def embed_video(self, vp, sig, op):
        cap = cv2.VideoCapture(vp)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        b_min, b_max = sig.min(), sig.max()
        norm = np.clip(((sig - b_min) / (b_max - b_min)) * 65535, 0, 65535).astype(np.uint16)
        out = cv2.VideoWriter(op, 0, fps, (w, h))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        f_idx = 0;
        d0, d1 = -self.delta / 4.0, self.delta / 4.0
        while True:
            ret, fr = cap.read();
            if not ret or f_idx >= len(norm): break
            yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)
            y = yuv[:, :, 0].astype(np.float32)
            bits = HammingECC.encode_16to28(norm[f_idx])
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                dct = cv2.dct(y[by:by + 16, bx:bx + 16])
                bit = bits[idx % 28]
                for cy, cx in self.coords:
                    c = dct[cy, cx]
                    # 绝对公平：不再乘以 tex_factor，直接用重装甲硬砸
                    dct[cy, cx] = np.round((c - d1) / self.delta) * self.delta + d1 if bit == 1 else np.round(
                        (c - d0) / self.delta) * self.delta + d0
                y[by:by + 16, bx:bx + 16] = cv2.idct(dct)
            yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR));
            f_idx += 1
        cap.release();
        out.release()
        return True, (b_min, b_max), fps

    def extract_video(self, vp):
        cap = cv2.VideoCapture(vp)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        extr = [];
        d0, d1 = -self.delta / 4.0, self.delta / 4.0
        while True:
            ret, fr = cap.read();
            if not ret: break
            y = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
            scores = np.zeros(28)
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                dct = cv2.dct(y[by:by + 16, bx:bx + 16])
                dist0 = sum(
                    abs(dct[cy, cx] - (np.round((dct[cy, cx] - d0) / self.delta) * self.delta + d0)) for cy, cx in
                    self.coords)
                dist1 = sum(
                    abs(dct[cy, cx] - (np.round((dct[cy, cx] - d1) / self.delta) * self.delta + d1)) for cy, cx in
                    self.coords)
                if dist1 < dist0:
                    scores[idx % 28] += 1
                elif dist0 < dist1:
                    scores[idx % 28] -= 1
            extr.append(HammingECC.decode_28to16([1 if s > 0 else 0 for s in scores]))
        cap.release()
        return np.array(extr)


# 2.3 对照组：扩频 SS (恢复绝对固定强度)
class SS_Watermarker(BaseBlindAdaptiveWatermarker):
    def __init__(self, border_blocks=3, alpha=25):
        super().__init__()
        self.alpha = alpha
        self.coords = [(1, 5), (5, 1), (2, 4), (4, 2), (1, 6), (6, 1), (3, 4), (4, 3), (2, 6), (6, 2), (3, 5), (5, 3),
                       (1, 8), (8, 1), (4, 5), (5, 4)]
        np.random.seed(42);
        base = [1] * 8 + [-1] * 8;
        np.random.shuffle(base);
        self.W = np.array(base)

    def embed_video(self, vp, sig, op):
        cap = cv2.VideoCapture(vp)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        b_min, b_max = sig.min(), sig.max()
        norm = np.clip(((sig - b_min) / (b_max - b_min)) * 65535, 0, 65535).astype(np.uint16)
        out = cv2.VideoWriter(op, 0, fps, (w, h))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        f_idx = 0
        while True:
            ret, fr = cap.read();
            if not ret or f_idx >= len(norm): break
            yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)
            y = yuv[:, :, 0].astype(np.float32)
            bits = HammingECC.encode_16to28(norm[f_idx])
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                dct = cv2.dct(y[by:by + 16, bx:bx + 16])
                b_val = 1 if bits[idx % 28] == 1 else -1
                for k, (cy, cx) in enumerate(self.coords): dct[cy, cx] += self.alpha * b_val * self.W[k]
                y[by:by + 16, bx:bx + 16] = cv2.idct(dct)
            yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR));
            f_idx += 1
        cap.release();
        out.release()
        return True, (b_min, b_max), fps

    def extract_video(self, vp):
        cap = cv2.VideoCapture(vp)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        extr = []
        while True:
            ret, fr = cap.read();
            if not ret: break
            y = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
            scores = np.zeros(28)
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                dct = cv2.dct(y[by:by + 16, bx:bx + 16])
                scores[idx % 28] += sum(dct[cy, cx] * self.W[k] for k, (cy, cx) in enumerate(self.coords))
            extr.append(HammingECC.decode_28to16([1 if s > 0 else 0 for s in scores]))
        cap.release()
        return np.array(extr)


# 2.4 对照组：STDM (恢复绝对固定强度)
class STDM_Watermarker(BaseBlindAdaptiveWatermarker):
    def __init__(self, border_blocks=3, delta=150):
        super().__init__()
        self.delta = delta
        self.coords = [(1, 5), (5, 1), (2, 4), (4, 2), (1, 6), (6, 1), (3, 4), (4, 3), (2, 6), (6, 2), (3, 5), (5, 3),
                       (1, 8), (8, 1), (4, 5), (5, 4)]
        np.random.seed(42);
        base = np.array([1] * 8 + [-1] * 8);
        np.random.shuffle(base);
        self.U = base / np.sqrt(16)

    def embed_video(self, vp, sig, op):
        cap = cv2.VideoCapture(vp)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        b_min, b_max = sig.min(), sig.max()
        norm = np.clip(((sig - b_min) / (b_max - b_min)) * 65535, 0, 65535).astype(np.uint16)
        out = cv2.VideoWriter(op, 0, fps, (w, h))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        f_idx = 0;
        d0, d1 = -self.delta / 4.0, self.delta / 4.0
        while True:
            ret, fr = cap.read();
            if not ret or f_idx >= len(norm): break
            yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)
            y = yuv[:, :, 0].astype(np.float32)
            bits = HammingECC.encode_16to28(norm[f_idx])
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                dct = cv2.dct(y[by:by + 16, bx:bx + 16])
                x = np.array([dct[cy, cx] for cy, cx in self.coords])
                x_proj = np.dot(x, self.U)
                y_proj = np.round((x_proj - d1) / self.delta) * self.delta + d1 if bits[idx % 28] == 1 else np.round(
                    (x_proj - d0) / self.delta) * self.delta + d0
                mod_x = x + (y_proj - x_proj) * self.U
                for k, (cy, cx) in enumerate(self.coords): dct[cy, cx] = mod_x[k]
                y[by:by + 16, bx:bx + 16] = cv2.idct(dct)
            yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR));
            f_idx += 1
        cap.release();
        out.release()
        return True, (b_min, b_max), fps

    def extract_video(self, vp):
        cap = cv2.VideoCapture(vp)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        extr = [];
        d0, d1 = -self.delta / 4.0, self.delta / 4.0
        while True:
            ret, fr = cap.read();
            if not ret: break
            y = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
            scores = np.zeros(28)
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                x = np.array([cv2.dct(y[by:by + 16, bx:bx + 16])[cy, cx] for cy, cx in self.coords])
                x_proj = np.dot(x, self.U)
                d_0, d_1 = abs(x_proj - (np.round((x_proj - d0) / self.delta) * self.delta + d0)), abs(
                    x_proj - (np.round((x_proj - d1) / self.delta) * self.delta + d1))
                if d_1 < d_0:
                    scores[idx % 28] += 1
                elif d_0 < d_1:
                    scores[idx % 28] -= 1
            extr.append(HammingECC.decode_28to16([1 if s > 0 else 0 for s in scores]))
        cap.release()
        return np.array(extr)


# 2.5 对照组：DWT-SVD (恢复绝对固定强度)
class DWT_SVD_Watermarker(BaseBlindAdaptiveWatermarker):
    def __init__(self, border_blocks=3, delta=60):
        super().__init__()
        self.delta = delta
        self.wavelet = 'haar'

    def embed_video(self, vp, sig, op):
        cap = cv2.VideoCapture(vp)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        b_min, b_max = sig.min(), sig.max()
        norm = np.clip(((sig - b_min) / (b_max - b_min)) * 65535, 0, 65535).astype(np.uint16)
        out = cv2.VideoWriter(op, 0, fps, (w, h))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        f_idx = 0;
        d0, d1 = -self.delta / 4.0, self.delta / 4.0
        while True:
            ret, fr = cap.read();
            if not ret or f_idx >= len(norm): break
            yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)
            y = yuv[:, :, 0].astype(np.float32)
            bits = HammingECC.encode_16to28(norm[f_idx])
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                block = y[by:by + 16, bx:bx + 16]
                LL, (HL, LH, HH) = pywt.dwt2(block, self.wavelet)
                U, S, Vh = np.linalg.svd(LL, full_matrices=False)
                S[0] = np.round((S[0] - d1) / self.delta) * self.delta + d1 if bits[idx % 28] == 1 else np.round(
                    (S[0] - d0) / self.delta) * self.delta + d0
                LL_mod = np.dot(U, np.dot(np.diag(S), Vh))
                y[by:by + 16, bx:bx + 16] = pywt.idwt2((LL_mod, (HL, LH, HH)), self.wavelet)
            yuv[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR));
            f_idx += 1
        cap.release();
        out.release()
        return True, (b_min, b_max), fps

    def extract_video(self, vp):
        cap = cv2.VideoCapture(vp)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        coords = self._get_fixed_lower_coords(h // self.block_size, w // self.block_size)
        extr = [];
        d0, d1 = -self.delta / 4.0, self.delta / 4.0
        while True:
            ret, fr = cap.read();
            if not ret: break
            y = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
            scores = np.zeros(28)
            for idx, (i, j) in enumerate(coords):
                by, bx = i * self.block_size, j * self.block_size
                block = y[by:by + 16, bx:bx + 16]
                LL, _ = pywt.dwt2(block, self.wavelet)
                _, S, _ = np.linalg.svd(LL, full_matrices=False)
                d_0, d_1 = abs(S[0] - (np.round((S[0] - d0) / self.delta) * self.delta + d0)), abs(
                    S[0] - (np.round((S[0] - d1) / self.delta) * self.delta + d1))
                if d_1 < d_0:
                    scores[idx % 28] += 1
                elif d_0 < d_1:
                    scores[idx % 28] -= 1
            extr.append(HammingECC.decode_28to16([1 if s > 0 else 0 for s in scores]))
        cap.release()
        return np.array(extr)


# ==================== 【3. 主控获取函数】 ====================
def get_base_pyvhr_data(video_path):
    print(f"[*] 正在利用 pyVHR 提取参考 BVP 信号 (Baseline)...")
    pipe = Pipeline()
    bvps, timesES, bpmES = pipe.run_on_video(video_path, winsize=12, roi_method='convexhull', method='cupy_CHROM',
                                             cuda=True, verb=False)
    bvp_base_raw = np.concatenate([b.flatten() for b in bvps])
    cap = cv2.VideoCapture(video_path)
    total_f, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return np.pad(bvp_base_raw, (0, max(0, total_f - len(bvp_base_raw))), 'edge')[:total_f], fps, total_f


def get_comp_pyvhr_data(video_path, total_frames, baseline_fps):
    pipe = Pipeline()
    bvps, _, _ = pipe.run_on_video(video_path, winsize=12, roi_method='convexhull', method='cupy_CHROM', cuda=True,
                                   verb=False)
    bvp_comp_raw = np.concatenate([b.flatten() for b in bvps])
    bvp_comp_raw = np.pad(bvp_comp_raw, (0, max(0, total_frames - len(bvp_comp_raw))), 'edge')[:total_frames]
    return filter_bvp_signal(bvp_comp_raw, baseline_fps)


def extract_subject_id(video_path):
    match = re.search(r'(subject\d+)', video_path, re.IGNORECASE)
    if match: return match.group(1)
    dir_name = os.path.basename(os.path.dirname(video_path))
    return dir_name if 'subject' in dir_name.lower() else 'unknown_subject'


# ==================== 【4. 大一统实验引擎】 ====================
if __name__ == "__main__":
    os.makedirs(OUT_BASE_DIR, exist_ok=True)
    qp_list = [22, 26, 30, 34, 38, 42]
    codecs_to_test = {'H.264': ('libx264', 'mp4'), 'H.265': ('libx265', 'mp4')}

    # 声明全部待测算法 (对照组均恢复固定装甲强度)
    algorithms = {
        'Proposed_JND': JND_DifferentialWatermarker(border_blocks=3),
        'QIM': QIM_Watermarker(border_blocks=3, delta=100),
        'SS': SS_Watermarker(border_blocks=3, alpha=25),
        'STDM': STDM_Watermarker(border_blocks=3, delta=150),
        'DWT_SVD': DWT_SVD_Watermarker(border_blocks=3, delta=60)
    }

    algo_plot_styles = {
        'Pure_Compression': {'marker': 'o', 'linestyle': '--', 'label': 'No Watermark (Pure Comp)'},
        'Proposed_JND': {'marker': '*', 'linestyle': '-', 'label': 'Proposed JND'},
        'QIM': {'marker': 's', 'linestyle': '-', 'label': 'QIM DCT'},
        'SS': {'marker': '^', 'linestyle': '-', 'label': 'Spread Spectrum'},
        'STDM': {'marker': 'D', 'linestyle': '-', 'label': 'STDM DCT'},
        'DWT_SVD': {'marker': 'v', 'linestyle': '-', 'label': 'DWT-SVD'}
    }

    for video_in in SUBJECT_VIDEO_LIST:
        subject_id = extract_subject_id(video_in)
        out_dir = os.path.join(OUT_BASE_DIR, subject_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n============================================================")
        print(f"[>>>] 开始全栈对比实验 (公平对抗+多数投票) | 目标: {subject_id}")
        print(f"============================================================")

        bvp_base_raw, baseline_fps, total_frames = get_base_pyvhr_data(video_in)
        bvp_baseline_filtered = filter_bvp_signal(bvp_base_raw, baseline_fps)

        metrics_data = {
            codec: {
                algo: {'MAE': [], 'SNR': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []}
                for algo in ['Pure_Compression'] + list(algorithms.keys())
            } for codec in codecs_to_test
        }
        init_visual_quality = {algo: {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0} for algo in algorithms.keys()}

        # --- 阶段 A：纯压缩 ---
        print(f"\n[阶段 A] 计算纯压缩影响基线 (Pure Compression)...")
        for codec_name, (encoder, ext) in codecs_to_test.items():
            for qp_val in qp_list:
                video_comp_only = os.path.join(out_dir, f"{subject_id}_direct_comp_{encoder}_QP{qp_val}.{ext}")
                print(f"  -> [Pure Comp] {codec_name} | QP={qp_val} ...", end="", flush=True)
                if compress_video_ffmpeg(video_in, video_comp_only, codec=encoder, qp_val=qp_val):
                    try:
                        bvp_comp_final = get_comp_pyvhr_data(video_comp_only, total_frames, baseline_fps)
                        mae, snr = SignalMetricsEvaluator.evaluate_metrics(bvp_baseline_filtered, bvp_comp_final,
                                                                           baseline_fps)
                        v_psnr, v_ssim, v_lpips = VisualMetricsEvaluator.evaluate_video_quality(video_in,
                                                                                                video_comp_only,
                                                                                                sample_interval=5)

                        metrics_data[codec_name]['Pure_Compression']['MAE'].append(mae)
                        metrics_data[codec_name]['Pure_Compression']['SNR'].append(snr)
                        metrics_data[codec_name]['Pure_Compression']['PSNR'].append(v_psnr)
                        metrics_data[codec_name]['Pure_Compression']['SSIM'].append(v_ssim)
                        metrics_data[codec_name]['Pure_Compression']['LPIPS'].append(v_lpips)
                        print(f" [OK]")
                    except Exception as e:
                        print(f" [Err: {e}]")
                        for met in ['MAE', 'SNR', 'PSNR', 'SSIM', 'LPIPS']:
                            metrics_data[codec_name]['Pure_Compression'][met].append(np.nan)
                else:
                    print(" [FFmpeg Fails]")
                    for met in ['MAE', 'SNR', 'PSNR', 'SSIM', 'LPIPS']: metrics_data[codec_name]['Pure_Compression'][
                        met].append(np.nan)

        # --- 阶段 B：水印测试 ---
        for algo_name, wm_instance in algorithms.items():
            print(f"\n[阶段 B] 正在评估水印算法: {algo_name}")
            video_embedded_avi = os.path.join(out_dir, f"{subject_id}_embedded_{algo_name}.avi")

            success, norm_params, fps_embed = wm_instance.embed_video(video_in, bvp_base_raw, video_embedded_avi)
            if not success:
                print(f"  [X] {algo_name} 嵌入失败，跳过。")
                continue
            b_min, b_max = norm_params

            avg_psnr, avg_ssim, avg_lpips = VisualMetricsEvaluator.evaluate_video_quality(video_in, video_embedded_avi,
                                                                                          sample_interval=1)
            init_visual_quality[algo_name] = {'PSNR': avg_psnr, 'SSIM': avg_ssim, 'LPIPS': avg_lpips}
            print(f"  -> 初始视觉质量 (PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f})")

            for codec_name, (encoder, ext) in codecs_to_test.items():
                for qp_val in qp_list:
                    video_comp_wm = os.path.join(out_dir,
                                                 f"{subject_id}_wm_{algo_name}_comp_{encoder}_QP{qp_val}.{ext}")
                    print(f"    -> [Attacking] {codec_name} | QP={qp_val} ...", end="", flush=True)
                    if compress_video_ffmpeg(video_embedded_avi, video_comp_wm, codec=encoder, qp_val=qp_val):

                        wm_extract_uint16 = wm_instance.extract_video(video_comp_wm)

                        if wm_extract_uint16 is not None and len(wm_extract_uint16) > 0:
                            raw_extracted_bvp = (wm_extract_uint16 / 65535.0) * (b_max - b_min) + b_min
                            final_bvp_filtered = filter_bvp_signal(raw_extracted_bvp, baseline_fps)
                            mae_wm, snr_wm = SignalMetricsEvaluator.evaluate_metrics(bvp_baseline_filtered,
                                                                                     final_bvp_filtered, baseline_fps)
                            v_psnr_wm, v_ssim_wm, v_lpips_wm = VisualMetricsEvaluator.evaluate_video_quality(video_in,
                                                                                                             video_comp_wm,
                                                                                                             sample_interval=5)

                            metrics_data[codec_name][algo_name]['MAE'].append(mae_wm)
                            metrics_data[codec_name][algo_name]['SNR'].append(snr_wm)
                            metrics_data[codec_name][algo_name]['PSNR'].append(v_psnr_wm)
                            metrics_data[codec_name][algo_name]['SSIM'].append(v_ssim_wm)
                            metrics_data[codec_name][algo_name]['LPIPS'].append(v_lpips_wm)
                            print(f" [OK] MAE: {mae_wm:.2f}")
                        else:
                            print(" [Extr Fails]")
                            for met in ['MAE', 'SNR', 'PSNR', 'SSIM', 'LPIPS']: metrics_data[codec_name][algo_name][
                                met].append(np.nan)
                    else:
                        print(" [FFmpeg Fails]")
                        for met in ['MAE', 'SNR', 'PSNR', 'SSIM', 'LPIPS']: metrics_data[codec_name][algo_name][
                            met].append(np.nan)

        # --- 阶段 C：融合绘图与 CSV 写入 ---
        print(f"\n[阶段 C] 正在为 {subject_id} 生成融合对比大图与数据表...")
        fig_sig, axs_sig = plt.subplots(2, 2, figsize=(18, 14))
        fig_sig.suptitle(f'Comprehensive Signal Robustness Comparison\nSubject: {subject_id}', fontsize=18,
                         fontweight='bold')

        for i, codec in enumerate(['H.264', 'H.265']):
            for algo, style in algo_plot_styles.items():
                if algo in metrics_data[codec] and len(metrics_data[codec][algo]['MAE']) > 0:
                    axs_sig[i, 0].plot(qp_list, metrics_data[codec][algo]['MAE'], marker=style['marker'],
                                       linestyle=style['linestyle'], linewidth=2.5, markersize=8, label=style['label'])
                    axs_sig[i, 1].plot(qp_list, metrics_data[codec][algo]['SNR'], marker=style['marker'],
                                       linestyle=style['linestyle'], linewidth=2.5, markersize=8, label=style['label'])

            axs_sig[i, 0].set_title(f'{codec} - Heart Rate MAE vs QP');
            axs_sig[i, 0].set_xlabel('Constant QP');
            axs_sig[i, 0].set_ylabel('MAE (BPM)');
            axs_sig[i, 0].legend();
            axs_sig[i, 0].grid(True, linestyle='--', alpha=0.7)
            axs_sig[i, 1].set_title(f'{codec} - Signal SNR vs QP');
            axs_sig[i, 1].set_xlabel('Constant QP');
            axs_sig[i, 1].set_ylabel('SNR (dB)');
            axs_sig[i, 1].legend();
            axs_sig[i, 1].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        plt.savefig(os.path.join(out_dir, f"{subject_id}_all_algos_signal_robustness.png"), dpi=300);
        plt.close(fig_sig)

        fig_vis, axs_vis = plt.subplots(2, 3, figsize=(24, 14))
        fig_vis.suptitle(f'Comprehensive Visual Quality Degradation\nSubject: {subject_id}', fontsize=18,
                         fontweight='bold')

        for i, codec in enumerate(['H.264', 'H.265']):
            for algo, style in algo_plot_styles.items():
                if algo in metrics_data[codec] and len(metrics_data[codec][algo]['PSNR']) > 0:
                    axs_vis[i, 0].plot(qp_list, metrics_data[codec][algo]['PSNR'], marker=style['marker'],
                                       linestyle=style['linestyle'], linewidth=2, label=style['label'])
                    axs_vis[i, 1].plot(qp_list, metrics_data[codec][algo]['SSIM'], marker=style['marker'],
                                       linestyle=style['linestyle'], linewidth=2, label=style['label'])
                    axs_vis[i, 2].plot(qp_list, metrics_data[codec][algo]['LPIPS'], marker=style['marker'],
                                       linestyle=style['linestyle'], linewidth=2, label=style['label'])

            axs_vis[i, 0].set_title(f'{codec} - PSNR vs QP');
            axs_vis[i, 0].set_xlabel('QP');
            axs_vis[i, 0].set_ylabel('PSNR (dB)');
            axs_vis[i, 0].legend();
            axs_vis[i, 0].grid(True, linestyle='--', alpha=0.7)
            axs_vis[i, 1].set_title(f'{codec} - SSIM vs QP');
            axs_vis[i, 1].set_xlabel('QP');
            axs_vis[i, 1].set_ylabel('SSIM');
            axs_vis[i, 1].legend();
            axs_vis[i, 1].grid(True, linestyle='--', alpha=0.7)
            axs_vis[i, 2].set_title(f'{codec} - LPIPS vs QP');
            axs_vis[i, 2].set_xlabel('QP');
            axs_vis[i, 2].set_ylabel('LPIPS');
            axs_vis[i, 2].legend();
            axs_vis[i, 2].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        plt.savefig(os.path.join(out_dir, f"{subject_id}_all_algos_visual_quality.png"), dpi=300);
        plt.close(fig_vis)

        csv_path = os.path.join(out_dir, f"{subject_id}_comprehensive_data.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metadata', 'Subject', subject_id, 'Baseline_FPS', baseline_fps])
            writer.writerow([])
            writer.writerow(['Algorithm_Initial_Visual_Quality', 'PSNR', 'SSIM', 'LPIPS'])
            for algo, vals in init_visual_quality.items():
                writer.writerow([algo, vals['PSNR'], vals['SSIM'], vals['LPIPS']])

            writer.writerow([])
            writer.writerow(['Codec', 'QP', 'Method', 'MAE_BPM', 'SNR_dB', 'PSNR_dB', 'SSIM', 'LPIPS'])
            for codec_name in codecs_to_test:
                for qp_idx, qp in enumerate(qp_list):
                    for algo in ['Pure_Compression'] + list(algorithms.keys()):
                        if qp_idx < len(metrics_data[codec_name][algo]['MAE']):
                            writer.writerow([
                                codec_name, qp, algo,
                                metrics_data[codec_name][algo]['MAE'][qp_idx],
                                metrics_data[codec_name][algo]['SNR'][qp_idx],
                                metrics_data[codec_name][algo]['PSNR'][qp_idx],
                                metrics_data[codec_name][algo]['SSIM'][qp_idx],
                                metrics_data[codec_name][algo]['LPIPS'][qp_idx]
                            ])

        print(f"[✔] {subject_id} 全部执行完毕。")
        print(f"============================================================\n")