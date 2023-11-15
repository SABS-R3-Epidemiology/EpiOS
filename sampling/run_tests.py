import os
import sys
import unittest
import argparse


def run_unit_tests():
    """
    Runs unit tests (without subprocesses).
    """
    tests = os.path.join(os.path.dirname(__file__),
                         'tests')
    suite = unittest.defaultTestLoader.discover(tests, pattern='test*.py')
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run unit tests for abm_model.',
        epilog='To run individual unit tests, use e.g.'
               ' $ abm_model/tests/dummy_test.py',
    )
    # Unit tests
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using `python` interpreter.',
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False

    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()
