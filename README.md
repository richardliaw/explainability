### Getting Started

```
git clone https://github.com/michaeltu1/lime.git
cd ~/lime
pip install -e .
git clone https://github.com/michaeltu1/Mask_RCNN.git
cd ~/Mask_RCNN
pip install -e .
pip install ray
git clone https://github.com/michaeltu1/explainability.git
```
  
### bandits_testing.ipynb
Using Epsilon Greedy Selection to choose impactful superpixels to perturb instead of choosing randomly  
The chosen superpixels are used to generate a data neighborhood for the Ridge Regression Models  

### coco.ipynb
Original Model: VGG16  
Dataset: MS COCO 2017 Validation Set  
Use Epsilon Greedy Selection + LIME to understand model during intermediate stages of training  

### explanations_during_training.ipynb
Show LIME explanation during training of InceptionV3 Keras Model

### lime_mask_rcnn.ipynb
1. Using LIME on the Bounding Box of Regions of Interest  
2. Comparing Image Segmentation (LIME Superpixels) vs. Object Segmentation (Mask-RCNN Regions of Interest)  

### superpixel_testing.ipynb
Visualizing explanations under small incremental transformations  
