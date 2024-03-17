# Object Detection System of Museum Works
Implementation of a system for the survey and classification of museum works.

## Dataset creation using PyTorch
Starting with a dataset of 22 images, i.e. Etruscan vases, they were divided into 5 classes, based on the type of vase. 
A set of transformations were applied to the images in the dataset in 4 different ways, using the torchvision package of the [PyTorch library](https://github.com/pytorch/pytorch), resulting in 88 images with dimensions 463x463x3:

| 1'Mode | 2'Mode | 3'Mode | 4'Mode |
| :---:   | :---: | :---: | :---:   |
| Resize, CenterCrop, ToTensor | RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ToTensor | Resize, RandomCrop, RandomVerticalFlip, GaussianBlur, ToTensor| Resize, CenterCrop, ColorJitter, ToTensor |

![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/9f1c2ac3-7e4b-4da1-a51c-4fe769149f26)

For details of my work, see: [Data Augmentation](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/tree/main/1.%20Data%20Augmentation) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pdf)

## Creating labels and obtaining the final dataset
Structuring the dataset by assigning labels to certain areas of the image, called bounding boxes, using the [Yolo_mark tool](https://github.com/AlexeyAB/Yolo_mark).

The file may be empty or it may contain one or more coordinates. Each coordinate is set as *'ID X Y WIDTH HEIGHT'*, where:
- ID: indicates the identification attributed to the different classes defined. In our case, five classes were defined between ID=0 and ID=4; 
- X: indicates the X co-ordinate of the centre of the object;
- Y: indicates the Y co-ordinate of the centre of the object;
- WIDTH: indicates the width of the object;
- HEIGHT: indicates the height of the object.

![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/b9d01bfd-acec-4c44-9f0a-74547b0ad9ba)

For details of my work, see: [Dataset YOLO format](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/tree/main/3.%20Training%26Testing%20Dataset%20with%20Detectron2/Dataset_YOLO_format) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pdf)

## Training and Testing
### K-fold Cross validation with Scikit-learn and PyTorch
Once the dataset was obtained, the k-fold cross validation algorithm was applied to analyse its accuracy. The implementation of the algorithm was done through the use of two libraries:

- [Scikit-learn](https://github.com/scikit-learn/scikit-learn), used to set the number of folds to be applied on the dataset. We chose k=5 as the number of folds, resulting in a split in which 80% refers to the training set, with 163 images, while the remaining 20% refers to the testing set, with 41 images;
- PyTorch, used for training the training set, with 500 epochs. At the end of the 500 epochs, the accuracy was evaluated using the testing set, containing the data that was not trained.

The training and evaluation process is carried out by alternating between 80% of the training set and 20% of the testing set each time, depending on the number of folds chosen via the Scikit-learn library. Once the algorithm has been run, the accuracy obtained is 99%.

For details of my work, see: [K-fold train Validation](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/2.%20Train%20K-fold%20Cross%20Validation/kfold_crossval_500epochs.ipynb) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pdf)

### Converting and splitting the dataset with Roboflow
[Detectron2](https://github.com/facebookresearch/detectron2) only supports datasets in COCO format, so labels were converted from YOLO TXT format to COCO JSON format), via [Roboflow](https://github.com/roboflow). Then the dataset in COCO format was divided to perform the training via Detectron2 in the following way:

<table><tbody>
<th valign="bottom">Set</th>
<th valign="bottom">Number of Images</th>
<tr><td align="left">Training Set</a></td>
<td align="center">145</td>
</tr>
<tr><td align="left">Validation Set</a></td>
<td align="center">34</td>
</tr>
<tr><td align="left">Testing Set</a></td>
<td align="center">25</td>
</tr>
</tbody></table>

For details of my work, see: [Dataset COCO format](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/tree/main/3.%20Training%26Testing%20Dataset%20with%20Detectron2/Dataset_COCO_format) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pdf) 

### Detectron2: Choice of model and backbone
Through the Detectron2 Model Zoo repository, the model pre-trained on the COCO dataset, Faster RCNN R-50 FPN 3X, was chosen, with its related architecture.
This backbone consists of a ResNet with 50 convolution levels and an FPN for feature extraction.

<table><tbody>
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<!-- ROW: faster_rcnn_R_50_FPN_3x -->
<tr><td align="left">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.209</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">40.2</td>
<td align="center">137849458</td>
</tr>
</tbody></table>

![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/3bc5c0cc-c0cd-4bd7-847b-94844d98f850)

For details of my work, see: [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pd) 

### Detectron2: Training Script Python
```
# Train Configuration using Detectron2 library
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
import os

cfg = get_cfg ()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS=4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.MAX_ITER = 500

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

cfg.TEST.EVAL_PERIOD = 100
os.makedirs(cfg.OUTPUT_DIR , exist_ok = True)
trainer = COCOEvaluator(cfg)
trainer.resume_or_load(resume = False)
trainer.train()
```
For details of my work, see: [DatasetVasi_maxiter500_cocoevaluator](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/3.%20Training%26Testing%20Dataset%20with%20Detectron2/DatasetVasi_maxiter500_prova1_cocoevaluator_personalizzata.ipynb) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pd) 

### Detectron2: Testing Script Python
```
# Test Configuration using Detectron2 library
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR ," model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("my_dataset_test", cfg , False , output_dir = "./output /")

val_loader = build_detection_test_loader(cfg ," my_dataset_test ")
inference_on_dataset(trainer.model , val_loader , evaluator)
```
For details of my work, see: [DatasetVasi_maxiter500_cocoevaluator](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/3.%20Training%26Testing%20Dataset%20with%20Detectron2/DatasetVasi_maxiter500_prova1_cocoevaluator_personalizzata.ipynb) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pd) 

### Detectron2: COCO Metrics in the training and testing phase

The model was evaluated using the "COCO metric with AP at IoU=.50:.05:.95", considering the IoU values relating to the bounding boxes, in which:

- Train: AP = 79.907%
- Test: AP = 81.760%

For details of my work, see: [DatasetVasi_maxiter500_cocoevaluator](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/3.%20Training%26Testing%20Dataset%20with%20Detectron2/DatasetVasi_maxiter500_prova1_cocoevaluator_personalizzata.ipynb) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pd) 

### Detectron2: Results
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/63889b9e-52cc-4b5b-a696-5dfc387864ce)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/4e537efd-6669-4ae1-80b4-d4f345c71d89)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/e7f16951-f34d-4071-b1c8-95d986e7671f)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/75884a68-e370-4e1b-8cef-a360d36bb32a)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/7187b00b-4f5f-482f-9ecd-aedc2e5f4895)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/35cfd308-4214-4f07-bdef-1df5e2c5934d)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/d32d380c-6a13-4d32-9e34-d058b4abd901)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/0e3f3c06-e1c0-4c02-beb1-58a54f3f2686)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/9829711c-60c6-4ef0-95da-ab6a4ad0e4f7)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/54b9d589-014c-41f6-9a73-981766d3f360)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/34b169fb-f87c-47ac-876b-f7060c721a37)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/0e96557f-7901-4f30-bffa-be0a53260add)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/79eaa297-6768-4937-9323-6e2074085a1a)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/78e44811-3517-42a8-986a-de2c2beb42f6)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/85219476-1775-4a52-9495-c43cffe33e43)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/f61ddb10-3081-4bb1-87d2-d20e8a527bd7)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/12060a50-aa6c-4f1d-886e-7a9142841bd9)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/ed917cf1-c7a5-4a9e-8b62-cb023e7ddb4f)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/1933bdbe-08cb-448e-985b-2d9017e69494)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/3edc30be-0781-4b87-88a5-be3deed45610)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/c2d70dd9-ec4b-4368-b65b-d5e2e82180c5)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/1df3ccbd-4a06-483f-82ac-de8624bed0e5)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/d5b803eb-1273-4b1e-b46a-e53b3c0af48f)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/77c93d6a-3b85-46a1-b5c8-cdade439a5a3)
![image](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/assets/105134422/b7516da3-b6d6-43f6-929a-7edae6452ba8)

For details of my work, see: [DatasetVasi_maxiter500_cocoevaluator](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/3.%20Training%26Testing%20Dataset%20with%20Detectron2/DatasetVasi_maxiter500_prova1_cocoevaluator_personalizzata.ipynb) and [thesis](https://github.com/giacomolat/Object-Detection-Sperimental-Thesis-for-Degree/blob/main/tesi.pd) 
