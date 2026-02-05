#!/usr/bin/env python
"""
Setup script for JUST_fiberassign package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Package metadata
NAME = 'JUST_fiberassign'
VERSION = '0.1.0'
DESCRIPTION = 'Fiber Assignment Package for JUST Telescope'
LONG_DESCRIPTION = read_readme()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
AUTHOR = 'JUST Telescope Team'
AUTHOR_EMAIL = 'just-telescope@example.com'
URL = 'https://github.com/just-telescope/JUST_fiberassign'
LICENSE = 'MIT'

# Required packages
INSTALL_REQUIRES = [
    'numpy>=1.19.0',
    'scipy>=1.7.0',
    'astropy>=4.0',
    'pandas>=1.3.0',
    'networkx>=2.6',
]

# Optional packages
EXTRAS_REQUIRE = {
    'visualization': ['matplotlib>=3.3.0'],
    'all': ['matplotlib>=3.3.0'],
}

# Package configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,

    # Package discovery
    packages=find_packages(),
    include_package_data=True,

    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # Python version requirements
    python_requires='>=3.7',

    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],

    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'just-fiber-assign=scripts.assign_targets:main',
        ],
    },

    # Additional files to include
    package_data={
        'JUST_fiberassign': [
            'survey_strategy/input/*.fits',
        ],
    },

    # Keywords
    keywords='astronomy telescope fiber spectroscopy assignment',

    # Project URLs
    project_urls={
        'Source': '',
        'Tracker': '',
    },
)