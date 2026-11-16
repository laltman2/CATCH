'''Test configuration: a stand-in for the optional yolov5 package.

The Localizer subclasses ``yolov5.YOLOv5``, which is optional, large and
needs trained weights, so the tests import it against this stub.
'''
import sys
import types

import pandas as pd


class FakeResults:

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self.frames = frames

    def pandas(self) -> 'FakeResults':
        return self

    @property
    def xyxy(self) -> list[pd.DataFrame]:
        return self.frames


class FakeYOLOv5:
    '''Records its arguments and returns the frames that a test sets.'''

    def __init__(self, model_path=None, device=None) -> None:
        self.model_path = model_path
        self.device = device
        self.frames: list[pd.DataFrame] = []
        self.calls: list[tuple] = []

    def predict(self, images, size=None) -> FakeResults:
        self.calls.append((images, size))
        return FakeResults(self.frames)


def make_fake_yolov5(version: str = '6.0.7') -> types.ModuleType:
    module = types.ModuleType('yolov5')
    module.__dict__.update(__version__=version, YOLOv5=FakeYOLOv5)
    return module


sys.modules['yolov5'] = make_fake_yolov5()
