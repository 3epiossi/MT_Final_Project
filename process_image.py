from module.extract_face import extract_face
from module.mosaic_image import mosaic_image
from module.overlay_images import overlay_person

def process_image(image, name):
    try:
        processed_image = extract_face(image)
        processed_image = mosaic_image(processed_image)
        processed_image = overlay_person(processed_image, name)
    except Exception as e:
        print(e)
    return processed_image