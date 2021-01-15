# from https://github.com/pytorch/ignite/blob/master/setup.py
import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read("README.md")

VERSION = find_version("package", "__init__.py")

extra = {}

install_requires = [
    "torch>=1.7.0",
    "pytorch-ignite>=0.4.2",
    "pytorch-lightning>=1.0.0",
    "prettytable>=2.0.0",
]

extra["testing"] = ["coverage>=5.3", "parameterized>=0.7.4"]
extra["quality"] = ["pylint", "isort[colors]", "black", "flake8"]

extra["dev"] = extra["testing"] + extra["quality"]


setup(
    name="toydet",
    version=VERSION,
    author="toyai",
    url="https://github.com/toyai/template",
    description="Just template",
    long_description_content_type="text/markdown",
    long_description=readme,
    license="MIT",
    python_requires=">=3.6.0",
    packages=find_packages(
        exclude=(
            "tests",
            "tests.*",
        )
    ),
    zip_safe=True,
    extras_require=extra,
    install_requires=install_requires,
)
