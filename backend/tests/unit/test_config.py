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


def test_storage_mode_defaults_to_file():
    s = Settings()
    assert s.storage_mode == "file"


def test_cosmos_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://localhost:8081")
    monkeypatch.setenv("COSMOS_KEY", "abc")
    s = Settings()
    assert s.cosmos_endpoint.startswith("https://")
    assert s.cosmos_key == "abc"


def test_is_cosmos_mode_property():
    s = Settings()
    assert s.is_cosmos_mode is False
    
    s2 = Settings(storage_mode="cosmos")
    assert s2.is_cosmos_mode is True
    
    s3 = Settings(storage_mode="COSMOS")
    assert s3.is_cosmos_mode is True
