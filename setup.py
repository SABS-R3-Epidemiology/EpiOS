#
# epios setuptools script
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the epios module.

    The easiest way would be to just ``import epios``, but note that this may
    fail if the dependencies have not been installed yet. Instead, we've put
    the version number in a simple version_info module, that we'll import here
    by temporarily adding the oxrse directory to the pythonpath using sys.path.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('epios'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md') as f:
        return f.read()


# Go!
setup(
    # Module name (lowercase)
    name='epios',

    # Version
    version=get_version(),

    description='Software for optimising sampling strategies using Epiabm output',

    long_description=get_readme(),

    license='BSD 3-Clause "New" or "Revised" License',

    # author='',

    # author_email='',

    maintainer='Tom Reed, Yunli Qi, Antonio Mastromarino',

    maintainer_email='thomas.reed@wolfson.ox.ac.uk, yunli.qi@dtc.ox.ac.uk, ' \
                     + 'antonio.mastromarino@wolfson.ox.ac.uk',

    url='https://github.com/SABS-R3-Epidemiology/EpiOS',

    # Packages to include
    packages=find_packages(include=('epios', 'epios.*')),

    # List of dependencies
    install_requires=[
        # Dependencies go here!
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'branchpro @ git+https://github.com/SABS-R3-Epidemiology/branchpro'
    ],
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)
