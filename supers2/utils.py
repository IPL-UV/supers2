import itertools
import rasterio
import pathlib

from rasterio.crs import CRS


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


def gdal_create(
    outfilename: str,
    dtype: str = "uint16",
    driver: str = "GTiff",
    count: int = 13,
    width: int = 5120,
    height: int = 5120,
    nodata: int = 65535,
    crs: int = 4326,
    affine: tuple = (-180, 0.5, 90, -0.5),
    **kwargs,
) -> pathlib.Path:
    """
    Fast creation of a new raster file using rasterio.

    Args:
        outfilename (str): Output filename.
        dtype (str): Data type of the raster.
        driver (str): GDAL driver to use.
        count (int): Number of bands in the raster.
        width (int): Width of the raster.
        height (int): Height of the raster.
        nodata (int): NoData value.
        crs (int): EPSG code of the raster.
        affine (tuple): Affine transformation of the raster.


    Returns:
        pathlib.Path: Path to the created raster file.
    """
    # Define the metadata for the new file
    meta = {
        "driver": driver,
        "dtype": dtype,
        "nodata": nodata,
        "width": width,
        "height": height,
        "count": count,
        "crs": CRS.from_epsg(crs),
        "transform": rasterio.transform.from_origin(*affine),
    }

    # Merge the metadata with the additional kwargs
    meta.update(kwargs)

    with rasterio.open(outfilename, "w", **meta) as dst:
        pass

    return pathlib.Path(outfilename)
