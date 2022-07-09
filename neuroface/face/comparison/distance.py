import torch


def l2_norm(embedding):
    """ L2 normalization technique implementation.
    
    Args:
        embedding (torch.tensor): Face image embedding represented as torch tensor.
    """
    
    return embedding / torch.sqrt(torch.sum(torch.multiply(embedding, embedding)))


def distance(embedding1, embedding2, distance_metric=0):
    """ Calculation of distance between detected faces embeddings.
    
    Args:
        embedding1 (torch.tensor): Original face image embedding represented as torch tensor.
        embedding2 (torch.tensor): Face image embedding to compare represented as torch tensor.
        distance_metric (int): Metric for evaluating the similarity of face embeddings - [0, 3].
            0 to select Euclidian distance, 1 to select Euclidian distance with L2 normalization,
            2 to select cosine similarity, 3 to select Manhattan distance. Default 0.
    """
    
    if distance_metric == 0:
        # Calculate euclidian distance.
        diff = torch.subtract(embedding1, embedding2)
        dist = torch.sum(torch.square(diff), 1)
    
    elif distance_metric == 1:
        # Calculate euclidian distance with L2 normalization.
        embedding1, embedding2 = l2_norm(embedding1), l2_norm(embedding2)
        diff = torch.subtract(embedding1, embedding2)
        dist = torch.sum(torch.square(diff), 1)
    
    elif distance_metric == 2:
        # Calculate distance based on cosine similarity.
        dot = torch.sum(torch.multiply(embedding1, embedding2), axis=1)
        norm = torch.linalg.norm(embedding1, axis=1) * torch.linalg.norm(embedding2, axis=1)
        similarity = dot / norm
        dist = torch.arccos(similarity) / torch.pi
    
    elif distance_metric == 3:
        # Calculate manhattan distance.
        diff = torch.subtract(embedding1, embedding2)
        dist = torch.sum(torch.abs(diff), 1)
    
    else:
        raise 'Undefined distance metric {}.'.format(distance_metric)
    
    return dist