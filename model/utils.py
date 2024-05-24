import os
import cv2
import tiffile
import numpy as np

VALID_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'bmp']
CLASSES = ['Cell']
colors = [(3,177,252)]

def read_lsm_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads lsm image and returns as array."""
    with tiffile.TiffFile(img_path) as tif:
        image = tif.pages[0].asarray()
    if np.transpose(image, (1, 2, 0)).shape[-1] == 1:
        return cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_GRAY2RGB)
    elif np.transpose(image, (1, 2, 0)).shape[-1] == 2:
        img = np.transpose(image, (1, 2, 0))
        stacked_array = np.dstack((img, np.zeros((512,512)).astype('uint8')))
        return stacked_array #cv2.cvtColor(stacked_array, cv2.COLOR_RGB2BGR)
    elif np.transpose(image, (1, 2, 0)).shape[-1] == 3:
        return np.transpose(image, (1, 2, 0)) #cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_RGB2BGR)
    else:
        img = np.transpose(image, (1, 2, 0))
        stacked_array = np.dstack((img[cell_channel], img[nuclei_channel], np.zeros((512,512)).astype('uint8')))
        return stacked_array #cv2.cvtColor(stacked_array, cv2.COLOR_RGB2BGR)

def read_standard_img(img_path):
    """Reads image in grayscale jpg/png/tif/bmp which contains cells only."""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,
                                     cv2.COLOR_GRAY2RGB))

def is_image_valid(img_path: str):
    """Checks if provided image is in correct format."""
    extension = img_path.split('.')[-1]
    if extension.lower() in VALID_IMAGE_EXTENSIONS:
        return True
    return False

def read_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads any possible image of cells (and/or nuclei)."""
    if img_path.endswith('lsm'):
        return read_lsm_img(img_path, cell_channel, nuclei_channel)
    elif is_image_valid(img_path):
        return read_standard_img(img_path)
    
def calculate_lsm(cell_counter, nuclei_counter,
                  img_path, cell_channel=0, nuclei_channel=1):
    """
    Calculates the resulting target values.
    Input params are:
    - img_path: path to lsm/jpg/png/tif/bmp image;
    - cell_channel: channel with cells. Default to 0;
    - nuclei_channel: channel with stained nuclei. Default to 1.

    Returns the result as a dictionary with the following fields:
    - Nuclei: count for stained nuclei detected (given lsm image only);
    - Cells: count for all the cells detected;
    - %: the target percentage for alive cells (given lsm image only).
    """
    img = read_lsm_img(img_path)

    cell_img = cv2.cvtColor(img[:,:,cell_channel], cv2.COLOR_GRAY2BGR)
    tmp_path = 'cell_tmp_img.png'
    cv2.imwrite(tmp_path, cell_img)
    cell_count = cell_counter.countCells(tmp_path)
    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        pass
    nuclei_count = nuclei_counter.countNuclei(img[:,:,nuclei_channel])
    percentage = (1 - nuclei_count/cell_count) * 100
    return {'Nuclei': nuclei_count, 'Cells': cell_count, '%': round(percentage,3)}

def calculate_standard(cell_counter, img_path):
    """
    Calculates cells only on given standard image.
    Input params are:
    - cell_counter: CellCounter class instance;
    - img_path: path to lsm/jpg/png/tif/bmp image.

    Returns the result as a dictionary with the following fields:
    - Nuclei: -100 (encoding for NaN);
    - Cells: count for all the cells detected;
    - %: -100 (encoding for NaN).
    """
    cell_count = cell_counter.countCells(img_path)
    return {'Nuclei': -100, 'Cells': cell_count, '%': -100}

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, draw_mode=0):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        # img_path (str): The path to input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[0]
    if img.shape[0] < 800:
        thickness = 1
    else:
        thickness = 2
    if draw_mode == 0:
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, thickness)
    else:
        img = cv2.circle(img, (x, y), 2, color, -1)