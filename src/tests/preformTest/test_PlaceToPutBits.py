import numpy as np
from sklearn.decomposition import PCA
from src.utils.utility_functions import find_least_important_components

# Example usage
latent_space_example = np.random.randn(100, 128)  # Simulating latent space (100 samples x 128 features)
num_bits = 33  # Number of bits to embed
least_important_indices, explained_variance = find_least_important_components(latent_space_example, num_bits)

print("Indices of least important components for embedding:")
print(least_important_indices)

# Display the variance of the selected components
print("\nVariance of selected components:")
for idx in least_important_indices:
    print(f"Component {idx + 1}: Variance = {explained_variance[idx]}")