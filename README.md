A Compact Convolutional Neural Network for Textured Surface Anomaly Detection

Surface defect detection is an essential task in the manufacturing process to ensure that the end product meets the quality standards and works in the way it is intended. Visual defect detection is done for steel surfaces, fabrics, wooden surfaces etc. This paper presents a compact convolutional neural architecture for weakly supervised textured surface anomaly detection for automating this task. The authors try to address the challenge of training from limited data and coarse annotations.

My article can be found at the following link:


https://towardsdatascience.com/a-compact-cnn-for-weakly-supervised-textured-surface-anomaly-detection-2572c3a65b80?sk=43dc0f494d5ff7985f2de2f1c9982e42


Keras and PyTorch implementation of the CompactCNN. 

Details of the paper can be found [here](https://www.semanticscholar.org/paper/A-Compact-Convolutional-Neural-Network-for-Textured-Racki-Tomazevic/17d3f90cb63fbac50a5e49b8a46e633ec1f526fd#extracted).

Requirements:
keras,tensorflow,opencv,matplotlib,os,numpy

To run the code use `python ./ccnn_pytorch/main.py` or `python ./ccnn_keras/main.py`

To create DAGM dataset in the form required for this repository run the following command.

`python DAGM_data_prep.py "dataset directory path where you want the folders to be created" "DAGM folder path"`

For example:  `python DAGM_data_prep.py "/john/doe/datasets" "john/doe/DAGM"`

I've included the images folder in the repository as well the weights of the trained models.

Following is a sample result of the network.

![Sample Result](https://github.com/msminhas93/CompactCNN/blob/master/SampleOutput/output17.png)

The network outputs an anomaly heatmap and an anomaly score. The segmentation model is trained for 25 epochs and classification model for 10 epochs. However, the best model based on validation loss is kept. 
