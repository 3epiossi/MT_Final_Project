import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from module.radian_background import create_image
import requests
from io import BytesIO


YELLOW_BGR = [132, 227, 255]  # BGR 格式的黃色
# YELLOW_BGR = [255, 255, 255]  # BGR 格式的黃色
# YELLOW_BGR = [0, 0, 0]  # BGR 格式的黃色
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

BACKGROUND_WIDTH = 1860
BACKGROUND_HEIGHT = 800
PERSON_TOP = 50
PERSON_HEIGHT = 700
PADDING = 30

SC_FONT_URL = "https://github.com/notofonts/noto-cjk/raw/refs/heads/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Medium.otf"
FONT_SIZE = 150

def load_font_from_url(url, font_size=60):
    """
    從遠端 URL 下載字型，並以指定字體大小載入為 Pillow 的字型物件 (ImageFont)。
    """
    response = requests.get(url)
    response.raise_for_status()  # 若下載失敗會丟出異常
    font_data = BytesIO(response.content)
    font = ImageFont.truetype(font_data, font_size)
    return font

def create_yellow_background(width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """
    創建指定尺寸 (width x height) 的黃色背景 (純色)。
    """
    background = np.full((height, width, 3), YELLOW_BGR, dtype=np.uint8)
    return background

def overlay_radian_background(overlay, blur_radius):
    """
    將疊加圖片進行邊緣羽化處理後疊加到漸層背景上 (由 create_image() 取得)。
    overlay: 必須含有 Alpha 通道 (RGBA)。
    blur_radius: 高斯模糊半徑，用於羽化邊緣。
    """
    if overlay.shape[2] != 4:
        raise ValueError("疊加圖片必須包含 Alpha 通道 (RGBA)")

    # 1) 取得漸層背景 (1900x800) 並轉為 BGR 格式
    background = create_image(BACKGROUND_WIDTH, BACKGROUND_HEIGHT) 
    background = np.array(background)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

    # 2) 計算貼上圖之縮放比例，確保高度符合 PERSON_HEIGHT
    scale = PERSON_HEIGHT / overlay.shape[0]
    new_width = int(overlay.shape[1] * scale)
    new_height = PERSON_HEIGHT

    # 3) 以最近鄰插值縮放疊加圖片，避免圖片模糊
    resized_overlay = cv2.resize(
        overlay, 
        (new_width, new_height), 
        interpolation=cv2.INTER_NEAREST
    )

    # 4) 分離疊加圖片的 BGR 與 Alpha 通道
    overlay_bgr = resized_overlay[:, :, :3]
    overlay_alpha = resized_overlay[:, :, 3] / 255.0  # 正規化到 [0, 1]

    # 5) 計算疊加範圍 (ROI)
    y_start = PERSON_TOP
    y_end = y_start + new_height
    x_start = (background.shape[1] - new_width) // 2
    x_end = x_start + new_width

    # 6) 對 Alpha 通道做高斯模糊，以羽化邊緣
    feathered_alpha = cv2.GaussianBlur(
        overlay_alpha,
        (blur_radius * 2 + 1, blur_radius * 2 + 1),
        blur_radius
    )
    # 7) 將羽化後的 Alpha 限制在 [0, 1]
    feathered_alpha = np.clip(feathered_alpha, 0, 1)

    # 8) 逐通道將疊加圖與背景混合
    roi_bgr = background[y_start:y_end, x_start:x_end, :].astype(np.float32)
    for c in range(3):
        roi_bgr[:, :, c] = roi_bgr[:, :, c] * (1 - feathered_alpha) + overlay_bgr[:, :, c] * feathered_alpha

    # 9) 更新 ROI 到背景
    background[y_start:y_end, x_start:x_end, :] = roi_bgr.astype(np.uint8)

    return background

def overlay_yellow_background(image):
    """
    功能 1 : 
    將原本 1860x800 的背景，置入新的 1920x1080 黃色底圖中，
    四周各留 30 px (上、下、左、右)，使最終輸出圖片維度為 1920x1080。
    假設參數 image 為尺寸 (800, 1860, 3)。
    """

    # 1) 建立 1920x1080 的黃色背景
    new_bg = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), YELLOW_BGR, dtype=np.uint8)

    # 2) 計算要貼上的位置 (ROI)
    #    - 在 1920x1080 的背景中，留 30 px 邊距
    y_start = PADDING
    y_end = y_start + BACKGROUND_HEIGHT  # 30 + 800 = 810
    x_start = PADDING
    x_end = x_start + BACKGROUND_WIDTH   # 30 + 1860 = 1890

    # 3) 將原本 1860x800 的背景貼到新的 ROI
    #    假設 image 大小正好是 (800, 1860, 3)
    new_bg[y_start:y_end, x_start:x_end] = image[0:BACKGROUND_HEIGHT, 0:BACKGROUND_WIDTH]

    return new_bg

def overlay_text(image, font, text=""):
    """
    使用 PIL 來在圖片上繪製中文文字。
    假設 image 為 1920x1080 大小的 BGR numpy 陣列。
    我們要在 y=830~1080 區域內，將文字上下左右置中顯示。
    """
    region_top = PADDING + BACKGROUND_HEIGHT
    region_bottom = IMAGE_HEIGHT - PADDING
    region_center_y = (region_top + region_bottom) // 2  # 955

    # 1) 先將 OpenCV 影像 (BGR) → PIL Image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 2) 取得繪圖物件
    draw = ImageDraw.Draw(image_pil)

    # 3) 設定顏色 (RGB)
    text_color = (255, 0, 0)  # 紅色 (注意是 PIL 的 RGB)

    # 4) 計算文字繪製位置，實現左右、上下置中
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (image.shape[1] - text_width) // 2
    # 記得 PIL 繪製文字的 (x,y) 是文字左上角，不是左下角
    text_y = (region_center_y - text_height // 2)

    # 5) 開始繪製文字
    draw.text(
        (text_x, text_y), 
        text, 
        font=font, 
        fill=text_color,
        stroke_width=1,               # 外框(描邊)寬度，可自行調整
        stroke_fill=text_color        # 若要整體更粗，就用同色
    )

    # 7) 再把 PIL Image (RGB) → OpenCV (BGR)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image

def overlay_person(image, name):
    font = load_font_from_url(SC_FONT_URL, font_size=FONT_SIZE)
    image = overlay_radian_background(image, 5)
    image = overlay_yellow_background(image)
    image = overlay_text(image, font, text=f'{name}是我们心中的红太阳')
    return image

if __name__ == '__main__':
    # 創建背景圖片
    # background = create_yellow_background(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    background = create_image(BACKGROUND_WIDTH, BACKGROUND_HEIGHT)
    background = np.array(background)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

    # 載入帶有 Alpha 通道的疊加圖片
    overlay = cv2.imread('test/robert_mosaiced.png', cv2.IMREAD_UNCHANGED)

    # 執行圖片疊加處理
    result = overlay_radian_background(background, overlay, blur_radius=5)

    # 顯示結果
    cv2.imwrite('result.png', result)
