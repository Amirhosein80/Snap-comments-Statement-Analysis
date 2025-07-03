from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="snapp_rating",
    version="0.1.0",
    author="Amirhossein Feiz",
    packages=find_packages(),
    install_requires=requirements,
)
