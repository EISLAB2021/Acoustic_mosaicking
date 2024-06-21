import kornia as K
import kornia.feature as KF
import cv2
import torch
import time
import kornia_moons.feature as KMF
import matplotlib.pyplot as plt
import numpy as np


def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img)
    return img


fname1 = ''  # acoustic image grayscale
fname2 = ''

img1 = load_torch_image(fname1)  # reference
img2 = load_torch_image(fname2)  # query

start_time = time.time()

## Loading weights
matcher = KF.LoFTR(pretrained='outdoor')

input_dict = {"image0": K.color.rgb_to_grayscale(img1),  # works on grayscale images
              "image1": K.color.rgb_to_grayscale(img2)}

with torch.no_grad():
    correspondences = matcher(input_dict)

'''
Now let’s clean-up the correspondences and estimate fundamental matrix between two acoustic images
'''
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()
end_time = time.time()

mkpts0_file = ''
np.savetxt(mkpts0_file, mkpts0, fmt='%.6f', delimiter=' ')

mkpts1_file = ''
np.savetxt(mkpts1_file, mkpts1, fmt='%.6f', delimiter=' ')

print(f"mkpts0 coordinates saved to {mkpts0_file}")
print(f"mkpts1 coordinates saved to {mkpts1_file}")

H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0) # RANSAC
# H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.0) #MAGSAC

inliers = inliers > 0

print('Local feature matching success rate (SR):', sum(p[0] for p in inliers / max(len(mkpts0), len(mkpts1))))
print('Correct match number:  {}'.format(sum(p[0] for p in inliers)))
print("Average time spent on a single match:%.2f  ms" % ((end_time - start_time) / len(mkpts1) * 1000))
KMF.draw_LAF_matches(
    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                 torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                 torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(mkpts1.shape[0]).view(1, -1, 1)),

    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),

    inliers,
    draw_dict={'inlier_color': (0.2, 1, 0.2),
               'tentative_color': None,
               'feature_color': (0.2, 0.5, 1),
               'vertical': False})

plt.axis('off')
plt.show()
