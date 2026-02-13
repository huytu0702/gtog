import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import Settings


def test_settings_yaml_path_defaults_to_settings_yaml():
    os.environ.pop("GRAPHRAG_SETTINGS_FILE", None)
    s = Settings()
    assert s.settings_yaml_path.name == "settings.yaml"


def test_settings_yaml_path_uses_env_override():
    os.environ["GRAPHRAG_SETTINGS_FILE"] = "./settings.cosmos-emulator.yaml"
    s = Settings()
    assert s.settings_yaml_path.name == "settings.cosmos-emulator.yaml"
    os.environ.pop("GRAPHRAG_SETTINGS_FILE", None)
