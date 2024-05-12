#Cell Calculator

## Brief description

**Cell Calculator** is an innovative application developed by project team "B" for processing mouse fibroblast cells of L929 cell line. It enables processing contrast microscopic images in *.lsm* format. Its basic functionality involves:
* counting the total number of cells presented on a given image;
* counting the number of stained nuclei presented on a given image;
* calculating the resulting percentage of alive cells presented on a given image.
To perform calculations, the application uses a unique model developed from scratch for this exactly project.

## Model design
The model can be divided into 2 separated submodels:
* Nuclei counter - counts the number of stained nuclei;
* Cell counter - counts the total number of cells.
The model obtains the results from each submodel, and then returns them as a dictionary which includes counts for nuclei, cells, and the resulting percentage.

### Nuclei counter design
The submodel for nuclei counting is based on classical computer vision algorithms used for image pre-processing and DBSCAN clustering algorithm used for differetiating between separated stained nuclei and counting them based on their spatial relations. The hyperparameter values for DBSCAN algorithm have been chosen by fine-tuning them on several images. The resulting nuclei counter can perform counting in approximately 3 seconds.

### Cell counter design
The submodel for cell counting is a YOLOV8 size-m object detection deep neural network which calculates cells by simply detecting them. It has been trained from scratch for 18 epochs on an object-detection dataset(more information is provided in **Data** section below). On the test object detection dataset the model showed Precision = 0.837 and Recall = 0.52. Thus, the model is clearly capable of detecting actual cells, but it cannot detect ALL the cells on the image. The explanation for this problem is simple: YOLOV8 models have an architectural limitation for the maximum number of objects that can be detected per image - in our case, that is 300 objects per image. To eliminate this issue we used the "regression on tales" approach - that is, we did predictions on separated parts of a single image and then combined them in the end. This technique allowed us to increase the limit up to 1,200 cells per image, which is already enough for the target images. Since YOLO models are designed to do real-time inference, regression on tales still works relatively fast, taking up to 10 seconds per image for inference.

### Model quality metrics
Below is a list of main model quality metrics. Unless specified, the metrics were obtained from the test subset of target data.

#### Nuclei counter
* MAE = 2.5;
* Inference time: 3 sec.

#### Cell counter
* Precision = 0.837*;
* Recall = 0.52*;
* MAPE = 0.12;
* Inference time: 10 sec.

**Metrics obtained on an object detection test subset.*

#### General model
* MAE (for target percentage): 0.018;
* Inference time: 13.5 sec.

## Data

<!-- TODO -->