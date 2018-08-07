### Getting Started

#### Install all dependencies:

1. git clone https://github.com/michaeltu1/lime.git
2. cd ~/lime
3. pip install -e .
4. git clone https://github.com/michaeltu1/Mask_RCNN.git
5. cd ~/mask_rcnn
6. pip install -e .
7. pip install ray
  
### bandits_testing.ipynb
Using Epsilon Greedy Selection to choose impactful superpixels to perturb instead of choosing randomly  
The chosen superpixels are used to generate a data neighborhood for the Ridge Regression Models  

### coco.ipynb
Original Model: VGG16  
Dataset: MS COCO 2017 Validation Set  
Use Epsilon Greedy Selection + LIME to understand model during intermediate stages of training  

### superpixel_testing.ipynb
Visualizing explanations under small incremental transformations  
