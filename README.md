# Cell Calculator

## Brief description

**Cell Calculator** is an innovative application developed by project team "B" for processing mouse fibroblast cells of L929 cell line. It enables processing contrast microscopic images in *.lsm* format, as well as images in standard formats, such as *jpeg*, *png* etc. Its basic functionality involves:
* counting the total number of cells presented on a given image;
* counting the number of stained nuclei presented on a given image (for *lsm* images only);
* calculating the resulting percentage of alive cells presented on a given image (for *lsm* images only).
To perform calculations, the application uses a unique model developed from scratch for this exactly project.

## Model design
The model can be divided into 2 separated submodels:
* Nuclei counter - counts the number of stained nuclei;
* Cell counter - counts the total number of cells.
The model obtains the results from each submodel, and then returns them as a dictionary which includes counts for nuclei, cells, and the resulting percentage.

### Nuclei counter design
The submodel for nuclei counting is based on classical computer vision algorithms used for image pre-processing and DBSCAN clustering algorithm used for differetiating between separated stained nuclei and counting them based on their spatial relations. The hyperparameter values for DBSCAN algorithm have been chosen by fine-tuning them on several images. The resulting nuclei counter can perform counting in approximately 3 seconds.

### Cell counter design
The submodel for cell counting is a YOLOv8-m object detection deep neural network which calculates cells by simply detecting them. It has been trained from scratch for 22 epochs on a third-party dataset(more information is provided in **Data** section below) in Google Colab cloud environment with default T4 GPU using Adam optimizer with default parameters and early stopping as a stopping criterion.

## Data

Original dataset which had been given to us was a set of unstandardized databases containing contrast images of L929 cells with some stained nuclei. Along with that, in response to our request we had been also given a set of images of cells only so that we could better analyze our model for cell counting.

Having performed EDA, it was clear that our data has several serious problems:
- Large data diversity (visually images differed significantly);
- Lack of data (fewer than 300 images available after filtering);
- No labels (no ground truth had been given to us - only the images).

As a result, it was decided to search for third-party datasets of cell microimages which would have visual appearance similar to ours. The dataset we found was LIVECell dataset, containing over 5,000 images (3,000+ training images), which was enough for us to train a deep model.

The test dataset for evaluating our model consists of 54 carefully chosen target images divided in 3 subsets so that images of different images could be analyzed in more details.

### Model quality metrics
Below is a list of main model quality metrics. The metrics have been measured on the target test dataset.

#### Nuclei counter
* MAE = 2.5.

#### Cell counter

|  | Precision | Recall | F1 score | MAPE | MAPE$_{manual}$ |
|---|---|---|---|---|
| Subset 1 | 0.957 | 0.899 | 0.926 | 0.146 | - |
| Subset 2 | 0.905 | 0.918 | 0.910 | 0.095 | - |
| Subset 3 | 0.989 | 0.923 | 0.954 | 0.102 | - |
| Micro Avg | 0.968 | 0.912 | 0.938 | 0.120 | 0.100 |
| Macro Avg | 0.950 | 0.913 | 0.930 | 0.114 | 0.080 |

#### General model
* MAE (for target percentage): 0.018;
* Inference time in the app: 4.5 sec.

## Run the code

Should you want to run the raw application code, follow the guidelines below:
1. Clone the repository using the prompt below:
```bash
git clone https://github.com/kikuroki/Cells-calculator.git
```
2. Enter the folder in which the repo is cloned and set up your environment. To do that, you need to have Python 3.7 or above pre-installed. Install the required dependency packages by running:
```bash
pip install -r requirements.txt
```
3. To start the application, execute the ```main_window.py``` file - for example, by running:
```bash
python main_window.py
```
4. Enjoy the application running!

<!-- TODO -->