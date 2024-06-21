import numpy as np


def read_points_from_txt(filename):
    """
    Reads 2D points from a text file.

    Parameters:
    filename (str): The path to the file containing points.

    Returns:
    list: A list of tuples representing the points (x, y).
    """
    with open(filename, 'r') as file:
        points = [tuple(map(float, line.strip().split())) for line in file]
    return points


# Read original coordinates on acoustic images
original_points_file = " "  # Replace with your coordinates file name
original_points = read_points_from_txt(original_points_file)

# Read the transformation matrix H
transformation_matrix_file = " "  # Replace with your transformation matrix file name
H = np.loadtxt(transformation_matrix_file)

# Apply the transformation matrix H to the original coordinates to get the new coordinates
transformed_points = []
for x, y in original_points:
    src_point = np.array([x, y, 1])
    transformed_point = H @ src_point
    transformed_point /= transformed_point[2]
    transformed_points.append(transformed_point[:2])

# Save the transformed coordinates to a text file
transformed_points_file = " "  # New coordinates file name
with open(transformed_points_file, 'w') as file:
    for point in transformed_points:
        file.write(f"{point[0]:.6f} {point[1]:.6f}\n")

print(f"Transformed points saved to {transformed_points_file}")
