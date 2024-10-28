import itertools


def define_iteration(dimension: tuple, chunk_size: int, overlap: int = 0):
    """
    Define the iteration strategy to walk through the image with an overlap.

    Args:
        dimension (tuple): Dimension of the S2 image.
        chunk_size (int): Size of the chunks.
        overlap (int): Size of the overlap between chunks.

    Returns:
        list: List of chunk coordinates.
    """
    dimy, dimx = dimension

    if chunk_size > max(dimx, dimy):
        return [(0, 0)]

    # Adjust step to create overlap
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap

    # Generate initial chunk positions
    iterchunks = list(itertools.product(range(0, dimy, y_step), range(0, dimx, x_step)))

    # Fix chunks at the edges to stay within bounds
    iterchunks_fixed = fix_lastchunk(
        iterchunks=iterchunks, s2dim=dimension, chunk_size=chunk_size
    )

    return iterchunks_fixed


def fix_lastchunk(iterchunks, s2dim, chunk_size):
    """
    Fix the last chunk of the overlay to ensure it aligns with image boundaries.

    Args:
        iterchunks (list): List of chunks created by itertools.product.
        s2dim (tuple): Dimension of the S2 images.
        chunk_size (int): Size of the chunks.

    Returns:
        list: List of adjusted chunk coordinates.
    """
    itercontainer = []

    for index_i, index_j in iterchunks:
        # Adjust if the chunk extends beyond bounds
        if index_i + chunk_size > s2dim[0]:
            index_i = max(s2dim[0] - chunk_size, 0)
        if index_j + chunk_size > s2dim[1]:
            index_j = max(s2dim[1] - chunk_size, 0)

        itercontainer.append((index_i, index_j))

    return itercontainer
