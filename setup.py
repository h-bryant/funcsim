import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="funcsim",
    version="0.0.9",
    license="bsd-3-clause",
    author="Henry Bryant",
    author_email="henry@tamu.edu",
    description="functional, simple stochastic simulation",
    keywords=["functional", "stochastic", "simulation"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/h-bryant/funcsim",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'six>=1.0.0'
    ],
    test_requires=[
        'scipy',
        'pytest',
        'sphinx'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
