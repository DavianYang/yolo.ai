#!/bin/bash

set -xu

black . --check

flake8 . --config .flake8

isort . --color --check

pylint package
