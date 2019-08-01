import argparse
import sys
import pytest
from src.model import test_model, train_model, tune_model


def main():
    parser = argparse.ArgumentParser(
        description="A command line tool to manage the project")
    parser.add_argument('stage',
                        metavar='stage',
                        type=str,
                        choices=['tune', 'train', 'test', 'unittest',
                                 'coverage', 'hypothesis'],
                        help="Stage to run. Either tune, train, test or unittest")

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    stage = parser.parse_args().stage

    if stage == 'tune':
        print('Tuning model...')
        tune_model()

    if stage == 'train':
        train_model(print_params=False)
        print('Model was saved')

    elif stage == 'test':
        test_model()

    elif stage == 'unittest':
        print("Unittesting model...")
        pytest.main(['-v', 'tests'])

    elif stage == 'coverage':
        print("Running coverage...")
        pytest.main(['--cov-report', 'term-missing', '--cov=src/', 'tests/'])

    elif stage == 'hypothesis':
        print("Running hypothesis...")
        pytest.main(['-v', '--hypothesis-show-statistics',
                     'tests/test_transformers_hypothesis.py'])


if __name__ == "__main__":
    main()
