from module.extract_face import extract_face
from module.mosaic_image import mosaic_image
from module.overlay_images import overlay_images

def process_image(image):
    try:
        processed_image = extract_face(image)
        processed_image = mosaic_image(processed_image)
        processed_image = overlay_images(processed_image)
    except Exception as e:
        print(e)
    return processed_image