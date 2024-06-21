import numpy as np
from numpy.linalg import inv


def read_points_from_txt(filename):
    with open(filename, 'r') as file:
        points = []
        for line in file:
            x, y = map(float, line.strip().split())
            points.append((x, y))
    return points


def calculate_homography_matrix(src_points, dst_points, max_reprojection_error):
    if len(src_points) != len(dst_points) or len(src_points) < 4:
        raise ValueError("At least 4 pairs of corresponding points are required to calculate the matrix H!")

    num_points = len(src_points)
    A = np.zeros((2 * num_points, 9))
    for i in range(num_points):
        x, y = src_points[i]
        u, v = dst_points[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    _, _, V = np.linalg.svd(A)
    h = V[-1] / V[-1, -1]
    H = h.reshape((3, 3))

    # Calculate the reprojection error
    reprojection_errors = []
    for i in range(num_points):
        src_point = np.array([src_points[i][0], src_points[i][1], 1])
        dst_point = np.array([dst_points[i][0], dst_points[i][1], 1])
        projected_point = H @ src_point
        projected_point /= projected_point[2]
        error = np.linalg.norm(projected_point - dst_point)
        reprojection_errors.append(error)

    # Check for abnormal points
    max_error = max(reprojection_errors)
    if max_error > max_reprojection_error:
        index_of_max_error = np.argmax(reprojection_errors)
        raise ValueError(
            f" the calculation deviation of the {index_of_max_error + 1} point is too large！The coordinate is {src_points[index_of_max_error]}")

    return H


src_points_file = ""  # Saved the corresponding points
dst_points_file = ""  # Saved the corresponding points

src_points = read_points_from_txt(src_points_file)
dst_points = read_points_from_txt(dst_points_file)

try:
    # Set the maximum reprojection error threshold
    max_reprojection_error = 5.0
    # Calculate the homography matrix H
    H = calculate_homography_matrix(src_points, dst_points, max_reprojection_error)
    print(H)

    # Save the transformation matrix H to a text file
    h_filename = "transformation_matrix_H.txt"
    np.savetxt(h_filename, H)
    print(f"Transformation matrix H saved to {h_filename}")
except ValueError as e:
    print(f"An error occurred：{e}")
