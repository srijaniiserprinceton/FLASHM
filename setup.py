""" :noindex:
Setup.py file with generic info
"""
import os
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as testcommand

# Utility function to read the README.md file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README.md file and 2) it's easier to type in the README.md file than to put a raw
# string in below ...


def read(fname):
    """From Wenjie Lei 2019"""
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as e:
        return "Can't open %s" % fname


long_description = "%s" % read("README.md")


# To run Pytest from the command line and run all necessary tests.
class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="FLASHM",
    description="Finite volume LAser SHockwave Modelling",
    long_description=long_description,
    version="0.0.1",
    author="Lucas Sawade",
    author_email="lsawade@princeton.edu",
    license='GNU Lesser General Public License, Version 3',
    keywords="Finite Volume, Shockwave, Laser",
    url='https://github.com/lsawade/FLASHM',
    packages=find_packages("."),
    package_dir={"": "."},
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    zip_safe=False,
    classifiers=[
        "Development Status :: Alpha",
        "Topic :: Utilities",
        "License :: GNU License",
    ],
    install_requires=[
        "numpy", "matplotlib", "flake8"
    ],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "tests": ["pytest", "py"]
    }
)
