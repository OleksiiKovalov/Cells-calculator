import os
import cv2
from ultralytics import YOLO

class CellCounter():
    """
    The class for object which performs cell counting.
    This is a part of the general model for obtaining target percentage.
    The objects of this class are not to be used explicitly - they function
    inside of the general Model class defined further.

    Input param is the path to pre-trained object detection model
    which calculates cells by detecting them using 'regression on
    tales' approach.

    Output value is the number of cells detected.
    """
    def __init__(self, path='model/best_m.pt'):
        self.model = YOLO(path)

    def countCells(self, img_path):
        """
        By calling this method, the CellCounter class instance calculates cells on a given image.
        The counting is done by using 'regression on tales' approach.
        The limit is 1,200 cells per image maximum (4 parts with 300 cells).

        The input param is the path to RGB image of cells.

        The output param is optimized count of cells.
        """

        cache_dir = 'model/cache'
        os.makedirs(cache_dir, exist_ok=True)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))

        img11 = img.copy()
        img11[255:,:,:] = 0
        img11[:,255:,:] = 0

        img12 = img.copy()
        img12[255:,:,:] = 0
        img12[:,:256,:] = 0

        img21 = img.copy()
        img21[:256,:,:] = 0
        img21[:,255:,:] = 0

        img22 = img.copy()
        img22[:256,:,:] = 0
        img22[:,:256,:] = 0

        img1112 = img.copy()
        img1112[255:,:,:] = 0

        img2122 = img.copy()
        img2122[:256,:,:] = 0

        img1121 = img.copy()
        img1121[:,255:,:] = 0

        img1222 = img.copy()
        img1222[:,:256,:] = 0

        cache_img_paths = ['img11.png', 'img12.png', 'img21.png', 'img22.png',
                        'img1112.png', 'img2122.png', 'img1121.png', 'img1222.png']

        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[0]), img11)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[1]), img12)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[2]), img21)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[3]), img22)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[4]), img1112)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[5]), img2122)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[6]), img1121)
        cv2.imwrite(os.path.join(cache_dir, cache_img_paths[7]), img1222)

        results = self.model([img_path, cache_dir+'/img11.png', cache_dir+'/img12.png',
                              cache_dir+'/img21.png', cache_dir+'/img22.png', cache_dir+'/img1112.png',
                              cache_dir+'/img2122.png', cache_dir+'/img1121.png', cache_dir+'/img1222.png'])  # return a list of Results objects

        res_values = {
            'img': 0,
            'img11': 0,
            'img12': 0,
            'img21': 0,
            'img22': 0,
            'img1112': 0,
            'img2122': 0,
            'img1121': 0,
            'img1222': 0
        }
        keys = list(res_values.keys())
        for i in range(len(results)):
            res_values[keys[i]] = (results[i].boxes.shape[0])
        res0 = res_values['img']
        res1 = res_values['img11'] + res_values['img12'] + res_values['img21'] + res_values['img22']
        res2 = res_values['img1112'] + res_values['img2122']
        res3 = res_values['img1121'] + res_values['img1222']

        for p in cache_img_paths:
            try:
                os.remove(os.path.join(cache_dir, p))
            except FileNotFoundError:
                pass
        os.rmdir(cache_dir)

        if res0 <= 290:
            return res0
        elif res2<= 590 and res3 <= 590:
            return int(round((res2 + res3) / 2))
        else:
            return res1
        