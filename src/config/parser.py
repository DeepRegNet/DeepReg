import yaml


def load(config_path):
    if config_path == "":
        config_path = "src/config/default.yaml"
    with open(config_path) as file:
        return yaml.load(file)


def save(config, out_dir):
    with open(out_dir + "/config.yaml", "w+") as f:
        f.write(yaml.dump(config))
