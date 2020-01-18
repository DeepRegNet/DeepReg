import yaml


def load_default():
    with open("src/config/default.yaml") as file:
        return yaml.load(file)
