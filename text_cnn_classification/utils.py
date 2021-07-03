import yaml


class AttrDict(dict):
    """Attr dict: make value private
    """

    def __init__(self, d):
        self.dict = d

    def __getattr__(self, attr):
        value = self.dict[attr]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value

    def __str__(self):
        return str(self.dict)


def load_config(config_file):
    """Load config file"""
    with open(config_file) as f:
        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = yaml.load(f)
    print(config)
    return AttrDict(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='text classification')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
