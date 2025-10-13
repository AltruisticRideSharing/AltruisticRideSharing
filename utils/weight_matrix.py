import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

def generate_weight_matrix(height, width):
    # Generate a random matrix with weights between 0.5 and 1
    return np.random.uniform(0.5, 1.0, size=(height, width))

# Generate the matrix and store it in a variable
weight_matrix = generate_weight_matrix(15, 15)

# Save the matrix to a file if needed (optional)
np.save("weight_matrix_15.npy", weight_matrix)

# Optionally, you can add a function to load this matrix
def load_weight_matrix():
    return np.load("weight_matrix.npy")