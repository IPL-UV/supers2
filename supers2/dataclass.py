import pathlib
from typing import Any, Dict, Optional, Union

import pydantic
import requests
import tqdm


class SRweights(pydantic.BaseModel):
    snippet: str
    path: Union[str, pathlib.Path, None] = None
    fullname: Optional[pathlib.Path] = None
    force_download: bool = False

    @pydantic.field_validator("path")
    def check_weights_path(cls, value):
        # If weights_path is None, we create a folder in the .config directory
        if value is None:
            value = pathlib.Path.home() / ".config" / "supers2"
            value.mkdir(parents=True, exist_ok=True)

        # Check if a valid directory was provided
        if pathlib.Path(value).is_dir():
            return value
        else:
            raise ValueError("weights_path must be a valid directory")

    @pydantic.model_validator(mode="after")
    def check_fullname(cls, values):
        fullpath = values.path / (values.snippet + ".pth")
        if not fullpath.exists() or values.force_download:
            print(f"File {fullpath} does not exist ... downloading")
            download_weights(fullpath)

        # Save the full path
        values.fullname = fullpath

        return values


class SRexperiment(pydantic.BaseModel):
    """This class is used to store the inference of a specific model"""

    fusionx2: Any
    fusionx4: Any
    srx4: Any

    def __repr__(self):
        message_fx2 = None
        message_fx4 = None
        message_srx4 = None

        if self.fusionx2 is not None:
            message_fx2 = f"'...'"
        if self.fusionx4 is not None:
            message_fx4 = f"'...'"
        if self.srx4 is not None:
            message_srx4 = f"'...'"

        return f"SRexperiment(fusionx2={message_fx2}, fusionx4={message_fx4}, srx4={message_srx4})"

    def __str__(self):
        return self.__repr__()


class AvailableModel(pydantic.BaseModel):
    """This class is used to define the structure of a specific model"""

    parameters: dict
    srclass: str


class AvailableModels(pydantic.BaseModel):
    object: Dict[str, AvailableModel]


def download_weights(model_snippet: pathlib.Path) -> pathlib.Path:
    """Download the weights of the model.

    Args:
        model_snippet (pathlib.Path): The path to the model snippet.

    Returns:
        pathlib.Path: The path to the downloaded weights.

    Raises:
        FileNotFoundError: If the file does not exist at the specified URL.
    """
    OFFICIAL_URL = "https://github.com/IPL-UV/supers2/releases/download/v0.1.0/"
    url = OFFICIAL_URL + model_snippet.name

    # Download the file directly
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # This will raise an HTTPError if the file does not exist
            with open(model_snippet, "wb") as f:
                for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        raise FileNotFoundError(f"Error downloading file from {url}: {e}")

    return model_snippet


# def download_weights(
#         total_size_in_bytes, filename_tmp, model_snippet, r_link):
 
#     OFFICIAL_URL = "https://github.com/IPL-UV/supers2/releases/download/v0.1.0/"
#     url = OFFICIAL_URL + model_snippet.name

#     # Download the file directly
#     try:
#         with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
#             with open(filename_tmp, 'wb') as f:
#                 for chunk in r_link.iter_content(chunk_size=8192):
#                         progress_bar.update(len(chunk))
#                         f.write(chunk)
#     except requests.exceptions.RequestException as e:
#         raise FileNotFoundError(f"Error downloading file from {url}: {e}")

#     return model_snippet