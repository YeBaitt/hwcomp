#!/usr/bin/env bash
set -euo pipefail

# DiffBIR 预训练权重下载脚本
# 下载 SD2.1 基础模型、IRControlNet v2.1 权重、Stage-1 SwinIR 权重 以及 NAFNet 权重
# 幂等：已存在的文件自动跳过

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIFFBIR_WEIGHTS="${ROOT_DIR}/vendor/DiffBIR/weights"
NAFNET_WEIGHTS="${ROOT_DIR}/vendor/NAFNet/weights"
HF_MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"

# 确保目标目录存在
mkdir -p "${DIFFBIR_WEIGHTS}" "${NAFNET_WEIGHTS}"

# ---------- DiffBIR 权重 ----------
# 下载函数：跳过已存在文件，支持 HF 镜像
download_if_missing() {
    local url="$1"
    local dest="$2"
    local fname
    fname="$(basename "$dest")"
    if [ -f "$dest" ]; then
        echo "[SKIP] ${fname} 已存在，跳过下载"
        return 0
    fi
    echo "[DOWNLOAD] ${fname} <- ${url}"
    # 若网址是 huggingface.co，替换为国内镜像
    if [[ "$url" == https://huggingface.co/* ]] && [ -n "${HF_MIRROR:-}" ]; then
        local mirror_url
        mirror_url="${url/https:\/\/huggingface.co/${HF_MIRROR}}"
        echo "  使用镜像: ${mirror_url}"
        wget -q --show-progress -O "$dest" "$mirror_url" || {
            echo "  镜像失败，回退原始地址..."
            wget -q --show-progress -O "$dest" "$url"
        }
    else
        wget -q --show-progress -O "$dest" "$url"
    fi
    echo "[OK] ${fname} 下载完成"
}

echo "=== DiffBIR v2.1 权重 ==="

# SD2.1 base (zsnr 版本，用于 v2.1)
download_if_missing \
    "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt" \
    "${DIFFBIR_WEIGHTS}/sd2.1-base-zsnr-laionaes5.ckpt"

# SD2.1 base (标准版本，用于 v1/v2)
download_if_missing \
    "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt" \
    "${DIFFBIR_WEIGHTS}/v2-1_512-ema-pruned.ckpt"

# IRControlNet v2.1
download_if_missing \
    "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt" \
    "${DIFFBIR_WEIGHTS}/DiffBIR_v2.1.pt"

# IRControlNet v2 (备用)
download_if_missing \
    "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth" \
    "${DIFFBIR_WEIGHTS}/v2.pth"

# Stage-1: SwinIR (Real-ESRGAN degradation, 用于 v2.1)
download_if_missing \
    "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth" \
    "${DIFFBIR_WEIGHTS}/realesrgan_s4_swinir_100k.pth"

echo ""
echo "=== DiffBIR v2.1 权重下载完毕 ==="
echo ""

# ---------- NAFNet 权重 ----------
# NAFNet-SIDD-width64 (图像去噪，宽度64，PSNR 40.30)
# 来源: megvii-research/NAFNet，通过 HF 镜像下载
echo "=== NAFNet 权重 ==="
NAFNET_SIDD_W64="${NAFNET_WEIGHTS}/NAFNet-SIDD-width64.pth"

download_if_missing \
    "https://huggingface.co/mikestealth/nafnet-models/resolve/main/NAFNet-SIDD-width64.pth" \
    "${NAFNET_SIDD_W64}"

# NAFNet-GoPro-width64 (图像去模糊，宽度64，PSNR 33.71)
NAFNET_GOPRO_W64="${NAFNET_WEIGHTS}/NAFNet-GoPro-width64.pth"

if [ -f "${NAFNET_GOPRO_W64}" ]; then
    echo "[SKIP] NAFNet-GoPro-width64.pth 已存在，跳过下载"
else
    echo "[INFO] NAFNet-GoPro-width64 需要从 Google Drive 下载:"
    echo "  URL: https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view"
    echo "  请手动下载后放入: ${NAFNET_GOPRO_W64}"
fi

echo ""
echo "=== 下载脚本完成 ==="
echo "DiffBIR 权重目录: ${DIFFBIR_WEIGHTS}"
echo "NAFNet 权重目录: ${NAFNET_WEIGHTS}"
ls -lh "${DIFFBIR_WEIGHTS}" 2>/dev/null || echo "(DiffBIR weights 目录为空)"
ls -lh "${NAFNET_WEIGHTS}" 2>/dev/null || echo "(NAFNet weights 目录为空)"
