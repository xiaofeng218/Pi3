import os
from pathlib import Path

from yacs.config import CfgNode as CN


def default_config() -> CN:
    cfg = CN(new_allowed=True)
    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.IMAGE_SIZE = 224
    cfg.EXTRA = CN(new_allowed=True)
    cfg.EXTRA.FOCAL_LENGTH = 5000
    cfg.MANO = CN(new_allowed=True)
    return cfg.clone()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | os.PathLike | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return str(resolved)
    return str(repo_root() / resolved)


def get_config(
    config_file: str,
    merge: bool = True,
    cache_dir: str = "./_DATA",
    update_cachedir: bool = False,
) -> CN:
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(resolve_repo_path(config_file))

    if update_cachedir:
        resolved_cache_dir = resolve_repo_path(cache_dir)

        def update_path(path: str) -> str:
            if os.path.isabs(path):
                return path
            return os.path.join(resolved_cache_dir, path)

        cfg.MANO.MODEL_PATH = update_path(cfg.MANO.MODEL_PATH)
        cfg.MANO.MEAN_PARAMS = update_path(cfg.MANO.MEAN_PARAMS)

    cfg.freeze()
    return cfg


def resolve_mano_path_template(path: str | None, data_dir: str | None = None) -> str | None:
    if path is None:
        return None
    resolved = str(path)
    if data_dir is not None and "${MANO.DATA_DIR}" in resolved:
        resolved = resolved.replace("${MANO.DATA_DIR}", str(data_dir))
    return resolved
