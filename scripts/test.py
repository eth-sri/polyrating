import unittest
import os

# Change the working directory to the parent directory of 'tests/'
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Discover all tests in the 'tests/' directory
test_suite = unittest.TestLoader().discover('tests', pattern='*.py')

# Run the discovered tests
test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)
