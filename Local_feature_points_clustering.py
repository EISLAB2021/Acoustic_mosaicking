import kornia as K
import kornia.feature as KF
import cv2
import torch
from sklearn.cluster import DBSCAN

import kornia_moons.feature as KMF
import matplotlib.pyplot as plt


def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img


# Load acoustic images
fname1 = ''
fname2 = ''
img1 = load_torch_image(fname1)
img2 = load_torch_image(fname2)

# Load features matcher
matcher = KF.LoFTR(pretrained='indoor')

# Convert images to grayscale
input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),
    "image1": K.color.rgb_to_grayscale(img2)
}

# Match features
with torch.no_grad():
    correspondences = matcher(input_dict)

# Get keypoints
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()

# Perform DBSCAN clustering on keypoints to identify dense regions
dbscan = DBSCAN(eps=5, min_samples=20)  # You can adjust these hyperparameters as needed
labels = dbscan.fit_predict(mkpts0)

# Find the label with the most keypoints (densest region)
max_label = max(set(labels), key=labels.tolist().count)

# Filter keypoints to keep only those in the densest region
filtered_mkpts0 = mkpts0[labels == max_label]
filtered_mkpts1 = mkpts1[labels == max_label]

# Estimate the Homography matrix using the filtered keypoints
H, inliers = cv2.findHomography(filtered_mkpts0, filtered_mkpts1, cv2.RANSAC, 5.0)

# Convert inliers to boolean array for visualization
inliers = inliers > 0

# Draw the matches using the filtered keypoints and inliers
KMF.draw_LAF_matches(
    KF.laf_from_center_scale_ori(torch.from_numpy(filtered_mkpts0).view(1, -1, 2),
                                 torch.ones(filtered_mkpts0.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(filtered_mkpts0.shape[0]).view(1, -1, 1)),

    KF.laf_from_center_scale_ori(torch.from_numpy(filtered_mkpts1).view(1, -1, 2),
                                 torch.ones(filtered_mkpts1.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(filtered_mkpts1.shape[0]).view(1, -1, 1)),

    torch.arange(filtered_mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),

    inliers,

    draw_dict={
        'inlier_color': (1, 1, 0),  # Yellow color for inliers
        'tentative_color': None,  # No tentative matches
        'feature_color': (0, 1, 0),  # Green color for features
        'vertical': True
    })

# Display the drawn image using matplotlib
plt.show()
