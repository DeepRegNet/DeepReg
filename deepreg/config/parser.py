import yaml


def load(config_path):
    with open(config_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save(config, out_dir):
    with open(out_dir + "/config.yaml", "w+") as f:
        f.write(yaml.dump(config))
