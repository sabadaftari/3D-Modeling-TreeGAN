import random
import torch
import matplotlib.pyplot as plt

def compute_diversity(generated_point_clouds: torch.Tensor, num_samples: int = 10) -> float:
    """
    Computes the diversity of generated samples by calculating the average Chamfer Distance 
    between pairs of randomly selected generated point clouds.
    
    Args:
        generated_point_clouds (torch.Tensor): A tensor containing the generated point clouds,
                                                 with shape [num_samples, num_points, 3].
        num_samples (int): The number of samples to use for diversity calculation.
        
    Returns:
        float: The average diversity score based on Chamfer Distance.
    """
    distances = []

    # Ensure we do not sample more indices than available
    num_generated = generated_point_clouds.size(0)
    if num_samples > num_generated:
        raise ValueError(f"num_samples ({num_samples}) cannot be greater than the number of generated point clouds ({num_generated}).")

    # Randomly sample indices for diversity calculation
    sample_indices = random.sample(range(num_generated), num_samples)
    
    # Compute Chamfer Distance for pairs of sampled point clouds
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            sample1 = generated_point_clouds[sample_indices[i]]
            sample2 = generated_point_clouds[sample_indices[j]]
            dist = ChamferDistance(sample1.unsqueeze(0), sample2.unsqueeze(0))  # Compute Chamfer Distance
            distances.append(dist.item())

    # Calculate the average diversity score
    diversity_score = sum(distances) / len(distances) if distances else 0.0
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

def ChamferDistance(x, y):
    """
    Compute Chamfer Distance between two point clouds x and y (no batch size).
    
    Args:
        x (torch.Tensor): Real point cloud of shape [num_points_x, 3].
        y (torch.Tensor): Generated point cloud of shape [num_points_y, 3].
    
    Returns:
        torch.Tensor: Chamfer Distance between point clouds x and y.
    """
    # Expand dimensions to prepare for broadcasting
    x_expand = x.unsqueeze(1)  # [num_points_x, 1, 3]
    y_expand = y.unsqueeze(0)  # [1, num_points_y, 3]
    
    # Calculate pairwise distances between points in x and points in y
    dist = torch.sum((x_expand - y_expand) ** 2, dim=-1)  # [num_points_x, num_points_y]
    
    # Compute the minimum distance from each point in x to the points in y
    dist_x_to_y = torch.min(dist, dim=1)[0]  # [num_points_x]
    
    # Compute the minimum distance from each point in y to the points in x
    dist_y_to_x = torch.min(dist, dim=0)[0]  # [num_points_y]
    
    # Return the average Chamfer Distance
    return torch.mean(dist_x_to_y) + torch.mean(dist_y_to_x)


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
    
# Visualization function for point clouds
def visualize_point_cloud(point_cloud, title="Point Cloud"):
    """
    Visualize a single point cloud.
    
    Args:
        point_cloud (np.array): Nx3 array of points.
        title (str): Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract X, Y, Z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    ax.scatter(x, y, z, c='b', marker='o')
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def visualize_batch_output_single_plot(generated_point_clouds):
    """
    Visualize the output of the generator for a given batch in a single 3D plot.

    Args:
        generator (nn.Module): The trained generator model.
        batch (torch.Tensor): The input batch containing real point clouds.
        device (torch.device): Device on which to perform computation (CPU/GPU).
        num_samples (int): Number of point clouds to visualize from the batch.
    """
    # Initialize a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Visualize all point clouds in the batch or a single point cloud
    if generated_point_clouds.size(0) > 1:
        for i in range(generated_point_clouds.size(0)):
            point_cloud = generated_point_clouds[i].cpu().numpy()

            # Ensure that the point cloud has the correct shape [num_points, 3]
            if point_cloud.ndim == 1:
                point_cloud = point_cloud.reshape(-1, 3)

            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], label=f"Sample {i + 1}", s=10)
    else:
        point_cloud = generated_point_clouds[0].cpu().numpy()

        # Ensure that the point cloud has the correct shape [num_points, 3]
        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(-1, 3)

        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], label="Generated Point Cloud", s=10)


    ax.set_title("Generated Point Clouds from Batch")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

    plt.show()