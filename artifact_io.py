import os
from datetime import datetime

import yaml


def ensure_parent_directory(path):
    parent_directory = os.path.dirname(path)
    if not parent_directory:
        raise ValueError(f'Output path must include a parent directory: {path}')
    os.makedirs(parent_directory, exist_ok=True)


def report_generated_at():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def save_yaml(path, values):
    if not isinstance(values, dict):
        raise TypeError('YAML report values must be a dictionary')
    if 'generated_at' in values:
        raise ValueError('YAML report values must not define generated_at')
    ensure_parent_directory(path)
    report_values = {'generated_at': report_generated_at(), **values}
    with open(path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(report_values, file, allow_unicode=True, sort_keys=False)


def save_report_figure(figure, path, **savefig_kwargs):
    if 'metadata' in savefig_kwargs:
        raise ValueError('Report figure metadata is managed by save_report_figure')
    ensure_parent_directory(path)
    figure.savefig(
        path,
        metadata={'Creation Time': report_generated_at()},
        **savefig_kwargs,
    )
