from setuptools import find_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="LPREpy",
    version="0.1",
    install_requires=[
            "pandas>=2.1.3",
    ],
    author="Julio C. R. Machado",
    author_email="julioromac@outlook.com",
    description="Library for internal contour development of liquid propellant rocket engines",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/juliomachad0/CEApy.git",
    packages=find_packages(include=['LPREpy']),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    #  package_data={'': ['cea-exec/*']},
)
