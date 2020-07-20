"""
This is a basic setuptools file.
"""

import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    """
    Unit test wrapper for the PyTest, including coverage repport
    """
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--color=yes', 'tests/'] #['--cov spectraplotpy tests/']
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
    name="pyErrorPred",
    version="0.0.1",
    packages=find_packages(),
    scripts=['ErrorPredictor.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        #"numpy", numpy is best installed via conda etc...
    ],
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.md'],
        '': ['samples/*'],
        '': ['models/*'],
        'pyErrorPred': ['data/*.txt', 'data/*.csv'],
    },
    zip_safe=False,
    # Project uses pytest for the tests

    tests_require=[
        'pytest',
        'pytest-cov',
        'mock'
    ],
    
    cmdclass={
        'test': PyTest
    },

    # metadata for upload to PyPI
    author="Naozumi Hiranuma",
    author_email="hiranumn@uw.edu",
    description="Tensorflow/Python implementation of local accuracy predictor for proteins ",
    license="None",
    keywords="Tensorflow Python proteins local accuracy",

    # project home page, if any
#    url=

    # could also include long_description, download_url, classifiers, etc.
)