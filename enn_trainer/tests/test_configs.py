from os import listdir
from pathlib import Path

import hyperstate
from hyperstate.schema.schema_change import Severity
from hyperstate.schema.schema_checker import SchemaChecker
from hyperstate.schema.types import load_schema

from enn_trainer.config import TrainConfig


def test_schema() -> None:
    schema_files = Path(__file__).parent.parent.parent / "config-schema.ron"
    path = str(schema_files)
    print(path)
    with open(path) as f:
        f.read()
    old = load_schema(path)
    checker = SchemaChecker(old, TrainConfig)
    if checker.severity() >= Severity.WARN:
        checker.print_report()
    assert checker.severity() == Severity.INFO


def test_configs() -> None:
    config_dir = Path(__file__).parent.parent.parent / "configs" / "entity-gym"
    for config_file in listdir(config_dir):
        print(config_file)
        hyperstate.load(TrainConfig, config_dir / config_file)
