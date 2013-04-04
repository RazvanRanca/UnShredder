#!/bin/sh

file=${1-"skeleton"}

pdflatex $file
bibtex $file
pdflatex $file
pdflatex $file

rm $file.aux
rm $file.bbl
rm $file.blg
rm $file.log
rm $file.toc
rm intro.aux
rm litRev.aux
rm probDef.aux
rm score.aux
rm search.aux

evince $file.pdf
