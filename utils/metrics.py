import numpy as np


def center_array(x, ax):
    return x - x.mean(axis=ax)

def cosine_similarity(pose1, pose2, _):
    """Compute the cosine similarity between two poses

    Args:
        pose1 (numpy array): first pose
        pose2 (numpy array): second pose
        _ : unused parameter

    Returns:
        float: the average cosine similarity over all joints
    """

    # Center the arrays
    pose1 = center_array(pose1, 0)
    pose2 = center_array(pose2, 0)
    # Compute norms
    norm1 = np.linalg.norm(pose1, axis=1)
    norm2 = np.linalg.norm(pose2, axis=1)
    # Compute dot products
    dot_products = np.sum(pose1 * pose2, axis=1)
    # Compute cosine similarity
    similarities = dot_products / (norm1 * norm2)

    return np.mean(similarities)

def inv_weighted_distance(pose1, pose2, keypoint_confs):
    """Compute the inverted weighted distance between two poses

    Args:
        pose1 (numpy array): first pose
        pose2 (numpy array): second pose
        keypoint_confs (numpy array): confidence scores

    Returns:
        float: the inverted weighted distance
    """

    # Center the arrays
    pose1 = center_array(pose1, 0)
    pose2 = center_array(pose2, 0)
    # Normalize
    pose1, pose2 = pose1 / np.linalg.norm(pose1), pose2 / np.linalg.norm(pose2)
    # Summation of weighted distances between keypoints
    total = 0
    for k in range(len(pose1)):
        total += keypoint_confs[k] * np.linalg.norm(pose1[k]-pose2[k])

    return 1 - total / keypoint_confs.sum()

def product(pose1, pose2, keypoint_confs):
    """Compute product of cosine similarity and inverted weighted distance
    between two poses

    Args:
        pose1 (numpy array): first pose
        pose2 (numpy array): second pose
        keypoint_confs (numpy array): confidence scores

    Returns:
        float: the product of cosine similarity and inverted weighted distance
    """

    score = cosine_similarity(pose1, pose2, keypoint_confs) * \
        inv_weighted_distance(pose1, pose2, keypoint_confs)

    return score

def moving_average(scores, sliding_window, batch_size):
    """Calculate moving average for batch of scores
    Args:
        scores (list): scores to calculate moving average
        sliding_window (int): sliding window
        batch_size (int): batch size
    Returns:
        list of floats: moving average for last batch of scores
    """
    window = np.ones(sliding_window) / sliding_window
    slicer = slice(sliding_window - 1, batch_size + sliding_window -1)

    if len(scores) == batch_size:  # First batch
        sliced_scores = np.pad(scores, sliding_window-1,
                            'edge')[:-sliding_window + 1]
    else:  # All the other batches
        sliced_scores = scores[-batch_size - sliding_window + 1:]
    
    return np.convolve(sliced_scores, window, mode='full')[slicer]