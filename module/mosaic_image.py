import cv2
import numpy as np

# 固定參數
HEIGHT = 700
HEIGHT_BLOCKS = 120
YELLOW_BGR_LIST = [132, 227, 255]  # BGR格式的皮膚色
RED_BGR_LIST = [0, 0, 255]         # BGR格式的紅色


def classify_red_and_skin(image):
    '''
    轉灰階並根據平均值分類紅色與皮膚色，保留透明區域
    Input: image (np.ndarray) (with alpha channel)
    Output: np.ndarray (with alpha channel)
    '''
    # 分離 BGRA 通道
    bgr = image[:, :, :3]   # 提取 BGR 通道
    alpha = image[:, :, 3]  # 提取 Alpha 通道
    
    # 轉換為灰階 (僅處理 BGR 部分)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # 計算灰階平均值 (僅對非透明區域計算)
    mean_value = np.mean(gray[alpha > 0])
    print(f"灰階平均值: {mean_value}")
    
    # 建立結果影像，初始大小與輸入相同
    result = np.zeros_like(image)
    # 保留透明度 (Alpha為0的部分維持透明)
    result[:, :, 3] = alpha
    
    # 根據灰階平均值，區分顏色
    # 低於或等於平均值 → 紅色
    result[gray <= mean_value, :3] = RED_BGR_LIST
    # 高於平均值 → 皮膚色
    result[gray > mean_value, :3] = YELLOW_BGR_LIST
    
    return result


def mosaic_image(image):
    '''
    將輸入圖片 (具透明通道)處理成：
    1) 高度固定為 HEIGHT px (同時依原圖比例調整寬度)
    2) 垂直方向分成 100 塊馬賽克 (每塊 6 px)
    3) 根據灰階值分類為紅色與皮膚色，並保留透明區域
    Input: image (np.ndarray) (with alpha channel)
    Output: np.ndarray (with alpha channel)
    '''
    # 取得原圖大小
    orig_h, orig_w, _ = image.shape
    
    # 1) 先將圖片等比例縮放至高度 HEIGHT px
    new_h = HEIGHT
    new_w = int(orig_w * new_h / orig_h) if orig_h != 0 else orig_w  # 避免除以零
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 2) 設定高度方向要分成 HEIGHT_BLOCKS 塊馬賽克 → 每塊馬賽克高 = HEIGHT / HEIGHT_BLOCKS (px)
    block_size_h = new_h // HEIGHT_BLOCKS  # 6 px
    # 若希望馬賽克保持正方形，可令 block_size_w = block_size_h
    # 或依照需求另計算 block_size_w
    block_size_w = block_size_h
    
    # 3) 先將圖片縮小至馬賽克數量大小
    mosaic_h = new_h // block_size_h  # HEIGHT / HEIGHT_BLOCKS
    mosaic_w = new_w // block_size_w  # 依照寬度大小計算
    
    # 4) 先縮小再放大
    mosaic = cv2.resize(resized_image, (mosaic_w, mosaic_h), interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 5) 使用前面定義的函式進行紅色 / 皮膚色分類
    result = classify_red_and_skin(mosaic)
    return result
