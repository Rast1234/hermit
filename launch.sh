#!/usr/bin/bash

rm output.*
python hermit.py
pdflatex output.tex
xdg-open output.pdf
