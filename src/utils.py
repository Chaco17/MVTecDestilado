"""Repository root, paths and composition of the Hydra config."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "conf"

# ${len:${data.categories}} in the YAML files: number of elements of a list
OmegaConf.register_new_resolver("len", len, replace=True)


def resolve_path(path):
    """Resolve a path of the config (relative to the repository root) to an absolute path."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(overrides=None):
    """Compose conf/config.yaml with optional overrides, e.g. ["+experiment=cnn_scratch/lr3e-4_bs32"].

    The compose API is the one Hydra recommends for Jupyter, where @hydra.main does not work.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides or []))
