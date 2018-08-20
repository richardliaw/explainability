### Getting Started

```
git clone https://github.com/michaeltu1/lime.git
cd lime
pip install -e .
cd ..

git clone https://github.com/michaeltu1/Mask_RCNN.git
cd Mask_RCNN
pip install -e .
cd ..

pip install ray

git clone https://github.com/michaeltu1/explainability.git
cd explainability

wget -qO- -O tmp.zip http://images.cocodataset.org/zips/val2017.zip && unzip tmp.zip && rm tmp.zip
wget -qO- -O tmp.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip tmp.zip && rm tmp.zip
cd ..

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
python setup.py install
cd ../..

pip install shapely

git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python setup.py install
```
  
### bandits_testing.ipynb
Using Epsilon Greedy Selection to choose impactful superpixels to perturb instead of choosing randomly  
The chosen superpixels are used to generate a data neighborhood for the Ridge Regression Models  

### coco.ipynb
Original Model: VGG16  
Dataset: [MS COCO](http://cocodataset.org/#home) 2017 Validation Set  
Use Epsilon Greedy Selection + LIME to understand model during intermediate stages of training  

### explanations_during_training.ipynb
Show LIME explanation during training of InceptionV3 Keras Model

### lime_mask_rcnn.ipynb
1. Using LIME on the Bounding Box of Regions of Interest  
2. Comparing Image Segmentation (LIME Superpixels) vs. Object Segmentation (Mask-RCNN Regions of Interest)  

### superpixel_testing.ipynb
Visualizing explanations under small incremental transformations  
