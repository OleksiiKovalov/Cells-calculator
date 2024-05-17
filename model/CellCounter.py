import os
import cv2
import numpy as np
#from ultralytics import YOLO

CLASSES = ['Cell']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

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
    def __init__(self, path):
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(path)

    def count(self, model, input_image):
        """
        Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.

        Returns:
            list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
        """
        # Load the ONNX model
    #     model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

        # Read the input image
        original_image: np.ndarray = cv2.imread(input_image)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 512

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(512, 512), swapRB=True)
        print('shape', blob.shape)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.2:  # originally >= .25
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
    #     result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.6)  # score, nms thresholds

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            # draw_bounding_box(
            #     original_image,
            #     class_ids[index],
            #     scores[index],
            #     round(box[0] * scale),
            #     round(box[1] * scale),
            #     round((box[0] + box[2]) * scale),
            #     round((box[1] + box[3]) * scale),
            # )

        # Display the image with bounding boxes
        # cv2.imshow("image", original_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return detections

    def countCells(self, img_path):
        """
        By calling this method, the CellCounter class instance calculates cells on a given image.
        The counting is done by using 'regression on tales' approach.
        The limit is 1,200 cells per image maximum (4 parts with 300 cells).

        The input param is the path to RGB image of cells.

        The output param is optimized count of cells.
        """

        return len(self.count(self.model, img_path))
        # cache_dir = 'model/cache'
        # os.makedirs(cache_dir, exist_ok=True)

        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (512, 512))

        # img11 = img.copy()
        # img11[255:,:,:] = 0
        # img11[:,255:,:] = 0

        # img12 = img.copy()
        # img12[255:,:,:] = 0
        # img12[:,:256,:] = 0

        # img21 = img.copy()
        # img21[:256,:,:] = 0
        # img21[:,255:,:] = 0

        # img22 = img.copy()
        # img22[:256,:,:] = 0
        # img22[:,:256,:] = 0

        # img1112 = img.copy()
        # img1112[255:,:,:] = 0

        # img2122 = img.copy()
        # img2122[:256,:,:] = 0

        # img1121 = img.copy()
        # img1121[:,255:,:] = 0

        # img1222 = img.copy()
        # img1222[:,:256,:] = 0

        # cache_img_paths = ['img11.png', 'img12.png', 'img21.png', 'img22.png',
        #                 'img1112.png', 'img2122.png', 'img1121.png', 'img1222.png']

        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[0]), img11)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[1]), img12)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[2]), img21)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[3]), img22)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[4]), img1112)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[5]), img2122)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[6]), img1121)
        # cv2.imwrite(os.path.join(cache_dir, cache_img_paths[7]), img1222)

        # results = self.model([img_path, cache_dir+'/img11.png', cache_dir+'/img12.png',
        #                       cache_dir+'/img21.png', cache_dir+'/img22.png', cache_dir+'/img1112.png',
        #                       cache_dir+'/img2122.png', cache_dir+'/img1121.png', cache_dir+'/img1222.png'])  # return a list of Results objects

        # res_values = {
        #     'img': 0,
        #     'img11': 0,
        #     'img12': 0,
        #     'img21': 0,
        #     'img22': 0,
        #     'img1112': 0,
        #     'img2122': 0,
        #     'img1121': 0,
        #     'img1222': 0
        # }
        # keys = list(res_values.keys())
        # for i in range(len(results)):
        #     res_values[keys[i]] = (results[i].boxes.shape[0])
        # res0 = res_values['img']
        # res1 = res_values['img11'] + res_values['img12'] + res_values['img21'] + res_values['img22']
        # res2 = res_values['img1112'] + res_values['img2122']
        # res3 = res_values['img1121'] + res_values['img1222']

        # for p in cache_img_paths:
        #     try:
        #         os.remove(os.path.join(cache_dir, p))
        #     except FileNotFoundError:
        #         pass
        # os.rmdir(cache_dir)

        # if res0 <= 290:
        #     return res0
        # elif res2<= 590 and res3 <= 590:
        #     return int(round((res2 + res3) / 2))
        # else:
        #     return res1
        