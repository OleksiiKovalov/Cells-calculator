# Cells Calculator

![[Header]](images/header.png)

## Quick references:
* [Brief description](#brief-description)
* [What's new?](#cellscalculatorv2-whats-new)
* [Get the CellsCalculator app](#get-the-cellscalculator-app)
* [Discover what our product is capable of](#discover-what-our-product-is-capable-of)
* [Available models](#available-models)
* [Data](#data)
* [Model quality metrics](#model-quality-metrics)
* [Run the code](#run-the-code)
* [See also](#see-also)
* [Contributors](#contributors)

## Brief description

**CellsCalculator** is an innovative application developed by project team "B" for processing mouse fibroblast cells of L929 cell line as well as cells of any other lines which visually appear to be similar to L929 line. It enables processing contrast microscopic images in *lsm* format, as well as images in standard formats, such as *jpeg*, *png* etc.

## CellsCalculatorV2: What's New?
The latest release v2.0 includes several major updates and innovations, which include:
* 2 different segmentation models were implemented for cell image processing, which enabled automatic extraction of information on cell morphology (averaged diameter, area, extrapolated volume);
* additional filtering configuration has been added for in-place filtering of irrelevant detections based on the cell area;
* processing of low-scale (x10 scale) images has been improved significantly using [SAHI](https://github.com/obss/sahi.git) approach;
* we have successfully created SOTA segmentation-based tracker of cellular spheroids, which enabled us to implement the functionality for processing frame sequences dsiplaying cellular spheroids;
* additional postprocessing utilities have been added for best user experience;
* folder-wise processing of cell microimages has been discontinued, based on user feedback

Summarizing, our achievements can be generally concluded as:
![[Releases]](images/Achievements.jpg)

On behalf of application architecture, starting from V2.0 **CellsCalculator** has plugin-based architecture, so that each independent tool is implemented within a dedicated plugin. As for now, there are 2 available plugins:
![[Releases]](images/plugins.png)

Visit [this](#discover-what-our-product-is-capable-of) section to see demo examples of the mentioned features.

## Get the CellsCalculator app

Should you want to try the CellsCalulator application, follow the guidelines below:
1. On project's GitHub main page, search for **Releases** section in the right menu;
![[Releases]](images/Screenshot_8.png)
2. Click on the latest release available;
3. In the opened release window, go to **Assets** section at the bottom and click on **CellsCalculator.zip** file archive;
![[Assets]](images/Screenshot_9.png)
4. Wait until the archive is completely loaded and then unpack it;
5. Enter the automatically created **CellsCalculator** folder and run the **main.exe** file.
6. Enjoy the application!

Remember: it is forbidden to rename or change location of ANY elements within the **CellsCalculator** directory, as it may badly influence the application behaviour, up to its full crash with unpredictable errors on the way. Please, note that the application does require the *model* folder with *best_m.onnx* file in it in order to calculate cells and work correctly in general.

## Discover what our product is capable of

**CellsCalculator** is a huge innovation indeed - just look at in what an awesome way it handles the job of calculating both cells and stained nuclei on given images, achieving nearly state-of-the-art performance on images of proper quality:

*Segmentation of cell instances (left: input image, right: processed image)*
![[cells_segmented]](images/Screenshot_39.jpg)

*Processing low-scale images (left: original model, right: SAHI+original model)*
![[cells_segmented_lowscale]](images/Screenshot_40.jpg)

*Tracking of cellular spheroids through a sequence of frames (left: visualization of segmentation results, right: plots representing time-series data of spheroids' morphology)*
![[spheroid_tracking]](images/spheroid_demo.gif)

## Available models
The application includes several models for different use-cases. Those are:
* 2 models for cell segmentation;
* 1 model for spheroid tracking;
* 1 model for nuclei counting.

### Nuclei counter model
The submodel for nuclei counting is based on classical computer vision algorithms used for image pre-processing and DBSCAN clustering algorithm used for differetiating between separated stained nuclei and counting them based on their spatial relations. Additionaly, some statistical-based methods are used for final filtering of the clusters obtained. The hyperparameter values for DBSCAN algorithm have been chosen by fine-tuning them on several images.
![[Nuclei_pipeline]](images/Screenshot_5.png)
If you want to take a look at how the model is designed and why certain elements were implemented, please visit our Colab notebook for more information: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FJm13fDHNh3m7r-yn-s7uWS98knm0uAF)

### Cell segmentation model
The model for cell segmentation is a YOLO11-x instance segmentation deep neural network by [ultralytics](https://docs.ultralytics.com) which enables high-quality low-resource-based cell segmentation. Below is its architecture:
![[YOLO]](images/YOLO_architecture.png)
The application includes 2 models which were trained with data of different resolution, which made their behaviours quite different. They have been trained from scratch for 59/74 epochs on a third-party dataset (more information is provided in [Data](#data) section below) in Kaggle cloud environment with default T100 GPU using Adam optimizer with default parameters and early stopping as a main stopping criterion.
If you want to take a look at how to train the model from scratch, please visit our dedicated [repository](https://github.com/EugenTheMachine/YOLOcfg.git) for more information.

## Data

### Microimages containing cells

Original dataset which had been given to us was a set of unstandardized databases containing contrast images of L929 cells with some stained nuclei. Along with that, in response to our request we had been also given a set of images of cells only so that we could better analyze our model for cell counting.
![[L929_images]](images/target_data_smaller.png)

After performing EDA, it became clear that our data had several serious problems:
- Large data diversity (visually images differed significantly);
- Lack of data (fewer than 300 images available after filtering);
- No labels (no ground truth had been given to us - only the images).

As a result, it was decided to search for third-party datasets of cell microimages which would have visual appearance similar to ours. The dataset we found was [LIVECell](https://sartorius-research.github.io/LIVECell/) dataset, containing over 5,000 images (3,000+ training images), which was enough for us to train a deep model.
![[LIVECell_images]](images/livecell_data_smaller.png)

The test dataset for evaluating our cell segmentation models consists of 94 carefully chosen target images divided into 7 subsets so that images of different images could be analyzed in more details. The images were labelled manually using CVAT web-tool.
![[L929_barchart]](images/barchart.png)

### Microimages containing cell stained nuclei

The test dataset for evaluating the stained nuclei counter model consists of 23 images, on which we could clearly differentiate between actual nuclei and noisy regions when creating ground truth labels.

It should be noted that many pictures have been filtered out due to different reasons, mostly - small cell scale or because of us being unable to infer any ground truth information ourselves due to the low quality of the data.

### Frame sequences containing spheroids

The dataset originally consisted of 52 sequences of frames, each ~42 frames long on average. It was labelled in semi-automatic mode using SAM-2 model and point prompts:
![[SAM_labelling]](images/SAM_labelling.png)
*(note: in some cases, bbox prompts had to be used instead to obtain better quality)*.

Next, it was splitted sequence-wise into train/val/test subsets, and then finally it was modified in accordance with ultralytics dataset requirements. For more information on dataset labelling procedure, check out our [dedicated repository](https://github.com/EugenTheMachine/SpheroidSAMLabelling.git).

## Model quality metrics
Below is a list of the main quality metrics of obtained models. We used the following metrics for evaluation of obtained cell segmentation models:
* **Precision** - represents the percentage of correctly identified instances in model outputs;
* **Recall** - represents the percentage of actual instances present on the image which were present in model outputs;
* **AP@50** - summarizes model behaviour, sort of averaging precision and recall, and gives the general understanding on how accurate the model is at identifying instances on images with minimal satisfying quality;
* **AP@50-95** - similar to AP@50, but evaluates the model more strictly, giving the general understanding on how accurate the model is at identifying instances on images as well as how accurate the model is at pixel-perfect detection of objects' borders.

For cellular spheroid tracker it was decided to evaluate just its segmentation model using the metrics listed above, as no tracking-specific labelling had been performed.

*Note: used quality metrics require calculation of IoU (intersection over union) ratio which, in case of segmentation models, can be obtained by using either object's bounding box or segmentation mask. That is why we will be providing metrics' values calculated for both bounding boxes (box) and masks (mask), which will be separated by `/` as `box/mask`.*

For evaluating our algorithm for stained cell nuclei counting, we used:
* **Mean Absolute Percentage Error (MAPE)**;
* **Mean Absolute Error (MAE)**;
* **Root Mean Squared Error (RMSE)**.

### YOLO11x-512 cell segmentation model

| Subset # | Precision | Recall | AP@50 | AP@50-95 | N Images |
|---|---|---|---|---|---|
| 0 | 85.7 / 87.5 | 88.5 / 90.4 | 87.5 / 89.7 | 43.1 / 45.2 | 12 |
| 1 | 80.0 / 60.0 | 0.2 / 0.2 | 40.2 / 30.2 | 17.2 / 14.1 | 6 |
| 2 | 83.3 / 86.1 | 0.6 / 0.6 | 42.0 / 43.4 | 21.1 / 23.2 | 14 |
| 3 | 64.8 / 66.2 | 75.3 / 77.8 | 70.2 / 72.7 | 32.2 / 36.0 | 3 |
| 4 | 78.8 / 81.5 | 72.9 / 75.3 | 72.5 / 75.9 | 39.7 / 41.6 | 41 |
| 5 | 84.0 / 81.3 | 2.2 / 2.2 | 42.9 / 41.6 | 22.6 / 24.1 | 13 |
| 6 | 27.5 / 48.2 | 14.2 / 25.0 | 17.4 / 33.2 | 4.4 / 9.6 | 5 |
| Macro Avg / Sum | 72.0 / 73.0 | 36.3 / 38.8 | 53.2 / 55.2 | 25.8 / 27.7 | 94 |
| Micro Avg / Sum | 78.0 / 79.3 | 46.7 / 48.6 | 60.7 / 62.7 | 31.4 / 33.3 | 94 |

### YOLO11x-680 cell segmentation model

| Subset # | Precision | Recall | AP@50 | AP@50-95 | N Images |
|---|---|---|---|---|---|
| 0 | 82.9 / 83.5 | 69.3 / 69.7 | 77.6 / 78.4 | 41.8 / 42.9 | 12 |
| 1 | 62.0 / 63.4 | 35.6 / 36.3 | 45.2 / 46.6 | 16.9 / 18.4 | 6 |
| 2 | 90.4 / 92.8 | 1.4 / 1.4 | 45.9 / 47.1 | 20.6 / 22.1 | 14 |
| 3 | 64.8 / 66.2 | 2.2 / 2.2 | 36.8 / 36.8 | 10.6 / 11.7 | 3 |
| 4 | 71.4 / 71.4 | 10.4 / 10.5 | 45.7 / 46.1 | 28.4 / 29.5 | 41 |
| 5 | 85.7 / 91.4 | 1.1 / 1.1 | 43.3 / 46.3 | 22.8 / 29.5 | 13 |
| 6 | 22.2 / 22.2 | 0.2 / 0.2 | 11.2 / 11.2 | 5.1 / 5.0 | 5 |
| Macro Avg / Sum | 71.0 / 72.6 | 17.2 / 17.4 | 43.7 / 44.6 | 20.9 / 22.7 | 94 |
| Micro Avg / Sum | 79.4 / 81.0 | 16.1 / 16.2 | 47.3 / 48.3 | 25.6 / 27.5 | 94 |

### Spheroid tracker YOLO11x-680 model

| - | Precision | Recall | AP@50 | AP@50-95 | N Images |
|---|---|---|---|---|---|
| Box | 94.4 | 91.9 | 95.1 | 87.1 | 451 |
| Mask | 95.7 | 93.2 | 96.1 | 97.1 | 451 |

### Stained Cell Nuclei counter
| MAPE | MAE | RMSE | N Images | N Nuclei |
|---|---|---|---|---|
| 0.059 | 1.0 | 1.629 | 23 | 325 |

## Run the code

Should you want to run the raw application code, follow the guidelines below. Note that the prompts are designed for Windows CMD - for bash you may need to use other syntax:
1. Clone the repository using the prompt below:
```bash
git clone https://github.com/kikuroki/Cells-calculator.git
```
2. Enter the folder in which the repo is cloned and set up your environment. To do that, you need to have Python 3.7 or above pre-installed. Install the required dependency packages by running:
```bash
pip install -r requirements.txt
```
3. To start the application, execute the ```main.py``` file - you can do that by running:
```bash
python main.py
```
4. Enjoy the application running!

## See also

If you are interested to study the project details more thoroughly, follow the links below to get more information on:
* [Model training guidelines](https://github.com/EugenTheMachine/YOLOcfg.git);
* [Used model configs & training artifacts](https://github.com/EugenTheMachine/ResultingModels.git);
* [Data labelling using SAM](https://github.com/EugenTheMachine/SpheroidSAMLabelling.git).

## Contributors

**CellsCalculator V2.0** has been developed by the following students of NTU "KhPI":
* **Ponomarov Y.** - *team-lead, ML engineer*;
* **Kuznesova I.** - *ML engineer, data labelling*;
* **Batiuchenko O.** - *software developer*;
* **Noskova K.** - *ML engineer*;
* **Glushchenko D.** - *lead of documentation editing assistance, lead of data labelling*;
* **Baluka A.** - *documentation editing assistance, data labelling*;
* **Ipatko K.** - *documentation editing assistance, data labelling*.
