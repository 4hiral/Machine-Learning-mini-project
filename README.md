***Road Damage Detection using Traffic Images***

***Objective***
The aim of this project is to detect and classify damages to roadway surfaces (e.g. cracks, potholes, and manholes) utilizing machine learning (SVM) and deep learning (CNN) classifiers applied to real-world images captured in traffic. The aim is to develop and implement an automated system for the assessment of roadway surface condition to aid municipal public works safety and other facets of infrastructure maintenance.

***Technologies & Frameworks***
 ***Programming Language:*** Python

 ***Frameworks and Libraries:*** TensorFlow · Keras · OpenCV · NumPy · Matplotlib · Scikit-learn

***Models Used***
 ***SVM (Support Vector Machine):*** A classical ML algorithm that classifies based on pixel-based feature vectors.

 ***CNN (Convolutional Neural Network):*** A deep learning model that is trained to learn features on the image directly (other models use specific features). 

 ***(Optional) YOLOv5:*** Used for object detection visualization of identified damages with bounding boxes.

***Datasets used***
 ***Road Damage Detection and Classification (Kaggle)***

   Utilized for Dataset-1 (/content/data1) (link: https://www.kaggle.com/datasets/alvarobasily/road-damage)
   Consists of labeled road images depicting different types of cracks and potholes.

 ***Road Damage Dataset – Potholes, Cracks, and Manholes (Kaggle)***

   Included in Dataset-2 (/content/data2) (link: https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
   Contains a variety of road-damage conditions captured in different environments, lighting, and camera angles.

***Workflow***
***Data Preprocessing***
       Images are Resize, normalize, and clean images.
       Create small, balanced subsets (work_ds1, work_ds2) for fast experimentation.

***Model Training***
       SVM is trained on flattened pixel vectors.
       CNN is trained on batches of images with data augmentation (rotation, flip, brightness).

***Model Evaluation***
       Metrics: Accuracy, Confusion Matrix, Cross-Dataset Performance.


***Results & Insights***
    | Model   | Dataset          | Accuracy    | Remarks                                            |
| ------- | ----------------- | ----------- | -------------------------------------------------- |
| **SVM** | Dataset-1 (/content/data1)  | **83.72%** | Demonstrates satisfactory performance for structured, clear images.         |
| **CNN** | Dataset-1 (/content/data1) | **91.25** | Better generalization when texture and illumination variations are present. |


***Key Insight***
  Deep learning (CNN) shows better generalization for road damage detection in real-world environments, while SVM is faster with small datasets.

***Conclusion***
  Developed an integrated machine learning + deep learning pipeline for automated identification of road defects. 
  
  Showcase of comparison of models and generalization across dataset. 
  
  Possibility for scaling for monitoring smart-city infrastructure using a camera-based approach.

Machine-Learning-mini-project/
│
├── 📄 README.md                        # This file gives an overall description of the project  
├── 📓 Road_Damage_Detection_SVM_CNN.ipynb   # Main Google Colab (Jupyter) document for the project  
│
├── 📂 data1/                           # Dataset for road damage detection and classification  
│   ├── train/                          # Folder for the training images  
│   ├── test/                           # Folder for testing images  
│   └── annotations/                    # Label file (if provided)  
│
├── 📂 data2/                           # Dataset for potholes, cracks, and manholes:  
│   ├── data/  
│   │   ├── images/                     # Images folder.  
│   │   └── labels/                     # Labels folder with file in text or XML formats.  
│
├── 📂 work_ds1/                        # Small portion of the dataset1 (sample) to help with training a model  
├── 📂 work_ds2/                        # Small portion of the dataset2 (sample) for cross-validation.  
│
├── 📂 yolov5/                          # YOLOv5 model files (if applied for detection visualization)  
│   ├── runs/                           # model outputs's predictions with bounding boxes  
│   └── weights/                        # YOLO model weights hv (yolov5s.pt, etc.)  
│
├── 📂 results/                         # model outputs's predictions, graphs, confusion matrix  or some results generated  
│   ├── cnn_confusion_matrix.png  
│   ├── svm_confusion_matrix.png  
│   └── model_comparison.png  
│
├── 📂 utils/                           # Helper scripts for data preprocessing and evaluation functions  
│   ├── preprocess.py  
│   ├── evaluate.py  
│   └── visualization.py  
│
└── 📂 reports/                         # reports generated from result, or inspection exports  
    └── Road_Damage_Report.pdf
