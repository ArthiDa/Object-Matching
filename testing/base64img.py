import cv2 as cv
import numpy as np
import base64
from io import BytesIO


# retrieve the image from the text file 
def get_image_from_text_file(file_name):
    with open(file_name, 'r') as f:
        imageDetails = f.read()
        starter = imageDetails.find(',')
        image_data = imageDetails[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        im = BytesIO(base64.b64decode(image_data))
        return im
    
if __name__ == '__main__':
    file_path = './base64.txt'
    img = get_image_from_text_file(file_path)
    img = cv.imdecode(np.frombuffer(img.read(), np.uint8), cv.IMREAD_UNCHANGED)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    