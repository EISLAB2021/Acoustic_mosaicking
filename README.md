# Acoustic_mosaicking
Acoustic Image Mosaicking Using Local Feature Matching

# Prerequisites
  Python 3.x
  Numpy
  OpenCV
  Kornia
  SciPy
  Scikit-learn 
  Matplotlib
  Pytorch > 1.3
  Pillow
  
# Usage
Utilities are defined in python scripts used as modules.Calculate the homography matrix H using the local feature matching results of adjacent overlapping acoustic images, and use the components of H for subsequent image mosaicking.
Acoustic image denoising is achieved by self-supervised denoising (first-stage), and enhanced by the fine features guided block.

# Acknowledgements
We borrowd and appreciate the contribution of code from the following sources：

Guided Image Filtering：https://kaiminghe.github.io/eccv10/index.html
Neighbor2Neighbor:https://github.com/TaoHuang2018/Neighbor2Neighbor
