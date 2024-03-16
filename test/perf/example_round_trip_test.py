import os
import filecmp
from glob import glob
from pathlib import Path

import pytest
import numpy as np

from midvoxio.writer import ArrayWriter
from midvoxio.parser import Parser
from midvoxio.vox import Vox
from midvoxio.models import default_palette

"""Round trip tests for official example files"""

@pytest.fixture(scope="session")
def out_path(tmp_path_factory: pytest.TempPathFactory):
    fn = tmp_path_factory.mktemp("tmp")
    yield fn

@pytest.fixture(scope="session")
def palette():
    palette1 = np.array(default_palette, np.uint32)
    palette = np.frombuffer(palette1.tobytes(), dtype=np.uint8).reshape(-1, 4)
    yield palette.tolist()

def pytest_generate_tests(metafunc):
    example_path = os.path.join(Path(__file__).parent, "data", "voxel-model", "vox", "*", "*.vox")
    filelist =  glob(example_path)
    filelist = [os.path.join(*f.split(os.sep)[-2:]) for f in filelist]
    metafunc.parametrize("localpath", filelist)

# test utilities

def example_round_trip(out_path, filepath):
    filename = os.path.split(filepath)[-1]
    out_filepath = os.path.join(out_path, filename)
    print(filepath, " -> ", out_filepath)
    
    # load data
    vox = Parser(filepath).parse()
    palette = vox.palettes[-1]
    data = vox.to_list(-1, -1)

    # write data
    writer = ArrayWriter(data, palette_arr=palette)
    writer.write(out_filepath)

    # verify output
    return filepath, out_filepath

def test_official_example_file(out_path: str, localpath):
    filepath = os.path.join(Path(__file__).parent, "data", "voxel-model", "vox", localpath)
    filepath, out_filepath = example_round_trip(out_path, filepath)
    assert filecmp.cmp(filepath, out_filepath)
