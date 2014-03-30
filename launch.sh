#!/usr/bin/bash

rm output.*
python2 hermit.py
pdflatex output.tex
xdg-open output.pdf
