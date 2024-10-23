import pathlib

import requests


def download_weights(model_snippet: pathlib.Path) -> pathlib.Path:
    """Download the weights of the model.

    Args:
        model_snippet (pathlib.Path): The path to the model snippet.

    Returns:
        pathlib.Path: The path to the downloaded weights.

    Raises:
        FileNotFoundError: If the file does not exist at the specified URL.
    """
    OFFICIAL_URL = (
        "https://github.com/JulioContrerasH/TFM_SR/releases/download/v.0.0.1/"
    )
    url = OFFICIAL_URL + model_snippet.name

    # Download the file directly
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # This will raise an HTTPError if the file does not exist
            with open(model_snippet, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        raise FileNotFoundError(f"Error downloading file from {url}: {e}")

    return model_snippet
