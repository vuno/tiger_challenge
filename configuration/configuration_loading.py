import yaml


def load_configuration(
    cfg_path: str,
) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    return cfg
