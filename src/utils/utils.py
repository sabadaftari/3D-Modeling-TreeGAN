import random
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ChamferDistance(x, y):
    """
    Compute Chamfer Distance between two point clouds x and y.
    
    Args:
        x (torch.Tensor): Real point cloud of shape [batch_size, num_points, 3].
        y (torch.Tensor): Generated point cloud of shape [batch_size, num_points, 3].
    
    Returns:
        torch.Tensor: Chamfer Distance between point clouds x and y.
    """
    # Ensure proper shapes
    if x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dimension
    if y.dim() == 2:
        y = y.unsqueeze(0)

    # Compute squared norms of x and y
    x_norm = torch.sum(x ** 2, dim=-1, keepdim=True)  # [batch_size, num_points_x, 1]
    y_norm = torch.sum(y ** 2, dim=-1, keepdim=True)  # [batch_size, num_points_y, 1]

    # Compute the pairwise dot products between points in x and points in y
    dot_product = torch.bmm(x, y.transpose(1, 2))  # [batch_size, num_points_x, num_points_y]

    # Compute pairwise distances: ||x||^2 + ||y||^2 - 2 * dot(x, y)
    dist = x_norm + y_norm.transpose(1, 2) - 2 * dot_product  # [batch_size, num_points_x, num_points_y]

    # Ensure non-negative distances (for numerical stability)
    dist = torch.clamp(dist, min=0.0)

    # Compute the minimum distance from each point in x to the points in y
    dist_x_to_y = torch.min(dist, dim=2)[0]  # [batch_size, num_points_x]

    # Compute the minimum distance from each point in y to the points in x
    dist_y_to_x = torch.min(dist, dim=1)[0]  # [batch_size, num_points_y]

    # Return the mean of the distances
    return torch.mean(dist_x_to_y) + torch.mean(dist_y_to_x)

def compute_diversity(generated_point_clouds, num_samples=10):
    """
    Computes the diversity of generated samples by calculating Chamfer Distance 
    between pairs of generated point clouds.
    
    Args:
        generated_point_clouds (torch.Tensor): Tensor of generated point clouds.
        num_samples (int): Number of samples to use for diversity calculation.
        
    Returns:
        diversity_score (float): Average diversity score based on Chamfer Distance.
    """
    distances = []
    sample_indices = random.sample(range(generated_point_clouds.size(0)), num_samples)
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            sample1 = generated_point_clouds[sample_indices[i]]
            sample2 = generated_point_clouds[sample_indices[j]]
            dist = ChamferDistance(sample1.unsqueeze(0), sample2.unsqueeze(0))  # Chamfer Distance between two point clouds
            distances.append(dist.item())

    diversity_score = sum(distances) / len(distances)
    return diversity_score

def visualize_combined_point_clouds(point_clouds):
    """
    Visualize all point clouds from the batch in a single plot.

    Args:
        point_clouds (tensor): Tensor of generated point clouds to visualize.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point cloud in the same plot
    for i, point_cloud in enumerate(point_clouds):
        # Ensure the point cloud has a compatible shape
        point_cloud = point_cloud.reshape(-1, 3).cpu()  # Move to CPU if on GPU
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], label=f"Sample {i + 1}")

    ax.set_title("Generated Point Clouds")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def compute_average_chamfer_loss(chamfer_distances):
    """
    Compute the average Chamfer Distance from a list of distances.

    Args:
        chamfer_distances (list): List of Chamfer Distance values for each batch.

    Returns:
        float: The average Chamfer Distance.
    """
    if len(chamfer_distances) == 0:
        return float('inf')
    
    avg_chamfer_loss = sum(chamfer_distances) / len(chamfer_distances)
    return avg_chamfer_loss
    
