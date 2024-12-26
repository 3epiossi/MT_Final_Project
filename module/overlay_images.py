import cv2
import numpy as np
from PIL import Image
from module.background import create_image

# YELLOW_BGR = [132, 227, 255]  # BGR 格式的黃色
YELLOW_BGR = [255, 255, 255]  # BGR 格式的黃色
# YELLOW_BGR = [0, 0, 0]  # BGR 格式的黃色
BACKGROUND_WIDTH = 1024
BACKGROUND_HEIGHT = 600
PERSON_TOP = 50
PERSON_HEIGHT = 400


def create_yellow_background(width=BACKGROUND_WIDTH, height=BACKGROUND_HEIGHT):
    """
    創建指定尺寸的黃色背景
    """
    background = np.full((height, width, 3), YELLOW_BGR, dtype=np.uint8)
    return background

def overlay_images_with_feathered_edges(background, overlay, blur_radius):
    """
    將疊加圖片進行邊緣羽化處理後疊加到背景上
    """
    if overlay.shape[2] != 4:
        raise ValueError("疊加圖片必須包含 Alpha 通道 (RGBA)")

    # 計算縮放比例
    scale = PERSON_HEIGHT / overlay.shape[0]
    new_width = int(overlay.shape[1] * scale)
    new_height = PERSON_HEIGHT

    # 使用最近鄰插值法縮放圖片，避免模糊
    resized_overlay = cv2.resize(overlay, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 分離疊加圖片的 BGR 和 Alpha 通道
    overlay_bgr = resized_overlay[:, :, :3]
    overlay_alpha = resized_overlay[:, :, 3] / 255.0  # 正規化到 [0, 1]

    # 計算疊加範圍
    y_start = PERSON_TOP
    y_end = y_start + new_height
    x_start = (background.shape[1] - new_width) // 2
    x_end = x_start + new_width

    # 使用高斯模糊對 Alpha 通道進行羽化處理
    feathered_alpha = cv2.GaussianBlur(overlay_alpha, (blur_radius * 2 + 1, blur_radius * 2 + 1), blur_radius)

    # 將羽化後的 Alpha 限制在 0 到 1 之間
    feathered_alpha = np.clip(feathered_alpha, 0, 1)

    # 將疊加圖片與背景圖片結合
    roi_bgr = background[y_start:y_end, x_start:x_end, :].astype(np.float32)
    for c in range(3):  # 對 BGR 每個通道分別進行混合
        roi_bgr[:, :, c] = roi_bgr[:, :, c] * (1 - feathered_alpha) + overlay_bgr[:, :, c] * feathered_alpha

    # 更新背景區域
    background[y_start:y_end, x_start:x_end, :] = roi_bgr.astype(np.uint8)
    return background

def overlay_images(image, blur_radius=5):
    """
    將疊加圖片進行邊緣羽化處理後疊加到背景上
    """
    # 創建背景圖片
    # background = create_yellow_background(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    background = create_image(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    background = np.array(background)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

    # 執行圖片疊加處理
    result = overlay_images_with_feathered_edges(background, image, blur_radius)

    return result

if __name__ == '__main__':
    # 創建背景圖片
    # background = create_yellow_background(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    background = create_image(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    background = np.array(background)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

    # 載入帶有 Alpha 通道的疊加圖片
    overlay = cv2.imread('test/robert_mosaiced.png', cv2.IMREAD_UNCHANGED)

    # 執行圖片疊加處理
    result = overlay_images_with_feathered_edges(background, overlay, blur_radius=5)

    # 顯示結果
    cv2.imwrite('result.png', result)
