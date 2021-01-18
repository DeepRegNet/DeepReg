import yaml

from deepreg.config.v011 import parse_v011


class TestV011:
    def test_grouped_mr_heart(self):
        with open("config/test/grouped_mr_heart_v011.yaml") as file:
            old_config = yaml.load(file, Loader=yaml.FullLoader)
        with open("demos/grouped_mr_heart/grouped_mr_heart.yaml") as file:
            latest_config = yaml.load(file, Loader=yaml.FullLoader)
        updated_config = parse_v011(old_config=old_config)
        assert updated_config == latest_config
