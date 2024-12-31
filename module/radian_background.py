import numpy as np
from PIL import Image
import math
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def gamma(x, gamma=2.6):
    return x ** (1 / gamma)
def create_image(width=800, height=800, red_white_gamma_index=2.6, center_white_gamma_index=0.9):
    angle_intensity = (gamma(np.random.rand(720), red_white_gamma_index)*255).astype(np.uint8)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    
    center_x = width // 2
    center_y = height // 2
    
    max_distance = ((width//2)**2 + (height//2)**2)**0.5
    blur_weights = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y
            
            distance = (dx**2 + dy**2)**(0.5)
            blur_weights[y, x] = min(1.0, gamma(distance / max_distance, center_white_gamma_index))

            angle_rad = math.atan2(dy, dx)
            
            angle_deg = int((angle_rad * 180 / math.pi + 360) % 360 * 2)
            
            intensity = angle_intensity[angle_deg]
            
            red, green, blue = 255, 255-intensity, 255-intensity
            output[y, x] = [red, green, blue]
    
    
    white = np.ones((3,), dtype=np.uint8) * 255
    for y in range(height):
        for x in range(width):
            blend_factor = blur_weights[y, x]
            output[y, x] = (white * (1 - blend_factor) + 
                          output[y, x] * blend_factor).astype(np.uint8)
    
    return Image.fromarray(output)

def save_image(image_path="radian_background.png", width=800, height=800, red_white_gamma_index=2.6, center_white_gamma_index=0.9):
    image = create_image(width, height, red_white_gamma_index, center_white_gamma_index)
    image.save(image_path)

if __name__ == "__main__":
    save_image("radian_background.png", 900, 600, 2.6, 0.9)