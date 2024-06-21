import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_saliency(image_path):
    # Read the acoustic image
    image = cv2.imread(image_path)

    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the luminance channel (L channel)
    L_channel = lab_image[:, :, 0]

    # Compute gradients of the L channel
    gradient_x = cv2.Sobel(L_channel, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(L_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Normalize gradient magnitude
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_8U)

    # Binarize the gradient image using Otsu's thresholding
    _, saliency = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return saliency


def generate_saliency_mask(saliency_map, threshold=128):
    # Threshold the saliency map
    binary_map = saliency_map

    # Perform morphological operations to fill holes and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
    opened_map = cv2.morphologyEx(closed_map, cv2.MORPH_OPEN, kernel)
    saliency_mask = cv2.erode(opened_map, kernel, iterations=1)

    return saliency_mask


if __name__ == "__main__":
    id1 = 1
    k = ''  # adjustable parameters
    for i in range(k):
        # Load raw acoustic image and compute saliency map
        image_path = " ".format(i + id1)
        raw_image = cv2.imread(image_path)
        saliency_map = compute_saliency(image_path)

        # Generate saliency mask
        saliency_mask = generate_saliency_mask(saliency_map)

        # Load first-stage denoised image
        guided_image = cv2.imread(' '.format(i + id1))

        # Ensure dimensions match
        if guided_image.shape[:2] != saliency_mask.shape:
            raise ValueError("Shape mismatch between guided_image and saliency_mask")

        # Prepare inputs for guided filter
        valid_indices = saliency_mask < guided_image.shape[0]
        guided_image_valid = guided_image[valid_indices]
        raw_image_valid = raw_image[valid_indices]

        # Apply guided filter
        radius = ''  # adjustable parameters
        eps = ''  # adjustable parameters
        imgGuidedFilter = cv2.ximgproc.guidedFilter(guided_image_valid, raw_image_valid, radius, eps)

        # Update guided filtered image
        Guide_filtered_image = raw_image.copy()
        Guide_filtered_image[valid_indices] = imgGuidedFilter

        # Final denoised result using weighted addition
        m = ''  # adjustable parameters
        n = ''  # adjustable parameters
        final_denoised_result = cv2.addWeighted(raw_image, m, Guide_filtered_image, n, 0)
        cv2.imwrite(' '.format(i + id1), final_denoised_result)

        # Display the results
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        axs[0, 0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Raw Acoustic Image")

        axs[0, 1].imshow(saliency_map, cmap='gray')
        axs[0, 1].set_title("Saliency Map")

        axs[0, 2].imshow(saliency_mask, cmap='gray')
        axs[0, 2].set_title("Saliency Mask")

        axs[1, 0].imshow(cv2.cvtColor(final_denoised_result, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("Final Denoised Result")

        axs[1, 1].imshow(cv2.cvtColor(guided_image, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("First-stage Denoised Image")

        axs[1, 2].imshow(cv2.cvtColor(Guide_filtered_image, cv2.COLOR_BGR2RGB))
        axs[1, 2].set_title("Guide Filtered Image")

        plt.tight_layout()
        plt.show()
