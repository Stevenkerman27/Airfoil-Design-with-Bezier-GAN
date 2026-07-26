import argparse
import os
import shutil

from alternating_finetuning import ALTERNATING_RESET_ARTIFACTS


def _validate_artifacts(artifacts):
    for name, path, is_directory in artifacts:
        if not isinstance(name, str) or not name:
            raise ValueError('Alternating reset artifact name must be a non-empty string')
        if not isinstance(path, str) or not path or not os.path.dirname(path):
            raise ValueError(f'Alternating reset artifact must have a parent directory: {path}')
        if not isinstance(is_directory, bool):
            raise ValueError(f'Alternating reset artifact directory flag must be boolean: {path}')


def existing_artifacts(artifacts=ALTERNATING_RESET_ARTIFACTS):
    _validate_artifacts(artifacts)
    return [(name, path, is_directory) for name, path, is_directory in artifacts if os.path.exists(path)]


def reset_alternating_finetuning(artifacts=ALTERNATING_RESET_ARTIFACTS):
    existing = existing_artifacts(artifacts)
    for _name, path, is_directory in existing:
        if is_directory:
            if not os.path.isdir(path):
                raise ValueError(f'Expected alternating reset directory at {path}')
            shutil.rmtree(path)
        else:
            if not os.path.isfile(path):
                raise ValueError(f'Expected alternating reset file at {path}')
            os.remove(path)
    return existing


def main():
    parser = argparse.ArgumentParser(
        description='Remove all alternating fine-tuning state so the next run starts at round 0.'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Perform the irreversible deletion. Without this flag, only list affected artifacts.',
    )
    arguments = parser.parse_args()
    artifacts = existing_artifacts()
    if artifacts:
        print('Alternating fine-tuning artifacts:')
        for name, path, _is_directory in artifacts:
            print(f'  {name}: {path}')
    else:
        print('No alternating fine-tuning artifacts exist.')
    if not arguments.confirm:
        print('Dry run only. Re-run with --confirm to delete these artifacts.')
        return
    deleted = reset_alternating_finetuning()
    print(f'Deleted {len(deleted)} alternating fine-tuning artifacts.')


if __name__ == '__main__':
    main()
