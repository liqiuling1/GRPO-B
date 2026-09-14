#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
latexmk -pvc -pdf -interaction=nonstopmode -synctex=1 -file-line-error main.tex
