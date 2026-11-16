'''Type stub for the parts of the optional yolov5 package that CATCH uses.'''
from os import PathLike
from typing import Any

__version__: str


class YOLOv5:
    def __init__(self,
                 model_path: str | PathLike[str],
                 device: str | None = None,
                 load_on_init: bool = True) -> None: ...

    def predict(self,
                image_list: Any,
                size: int = 640,
                augment: bool = False) -> Any: ...
