import copy

import pytest

from poeta.config import ConfigError, _validate, load_config


@pytest.fixture(scope="module")
def base():
    return load_config()


@pytest.mark.parametrize("hud", [{"corner": "centro"}, {"size": 5}, {"size": True}, {"opacity": 0.01},
                                 {"margin": -1}, {"click_through": "si"}, {"hide_when_idle": 1}])
def test_bad_hud_config_is_rejected(base, hud):
    cfg = copy.deepcopy(base)
    cfg["ui"]["hud"] = hud
    with pytest.raises(ConfigError, match="hud"):
        _validate(cfg)


def test_default_hud_config_is_valid(base):
    _validate(copy.deepcopy(base))
