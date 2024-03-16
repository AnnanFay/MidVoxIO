import os
import filecmp
from glob import glob
from pathlib import Path

import pytest
import numpy as np

from midvoxio.voxio import write_list_to_vox, vox_to_arr
from midvoxio.models import default_palette

"""Round trip tests simply test loading and writing the same data.

Failure if output doesn't match input, or if it breaks a target time threshold."""

@pytest.fixture(scope="session")
def out_path(tmp_path_factory: pytest.TempPathFactory):
    fn = tmp_path_factory.mktemp("tmp")
    yield fn

@pytest.fixture(scope="session")
def palette():
    palette1 = np.array(default_palette, np.uint32)
    palette = np.frombuffer(palette1.tobytes(), dtype=np.uint8).reshape(-1, 4)
    yield palette.tolist()

# test utilities

def round_trip(out_path, filename, **kwargs):
    filepath = os.path.join(Path(__file__).parent, "data", filename)
    out_filepath = os.path.join(out_path, filename)
    print(filepath, " -> ", out_filepath)

    vox = vox_to_arr(filepath)
    write_list_to_vox(vox, out_filepath, **kwargs)

    return filepath, out_filepath

def terrain_round_trip(out_path, filename):
    palette_path = os.path.join(Path(__file__).parent, "data", "simple_terrain_palette.png")
    return round_trip(out_path, filename, palette_path=palette_path)

# tests

def test_empty_file(out_path: str, palette):
    filepath, out_filepath = round_trip(out_path, "empty.vox", palette_arr=palette)
    assert filecmp.cmp(filepath, out_filepath)

def test_default_file(out_path: str, palette):
    filepath, out_filepath = round_trip(out_path, "default.vox", palette_arr=palette)
    assert filecmp.cmp(filepath, out_filepath)

def test_small_terrain(out_path: str):
    filepath, out_filepath = terrain_round_trip(out_path, "small_random_terrain.vox")
    assert filecmp.cmp(filepath, out_filepath)

def test_medium_terrain(out_path: str):
    filepath, out_filepath = terrain_round_trip(out_path, "medium_random_terrain.vox")
    assert filecmp.cmp(filepath, out_filepath)
