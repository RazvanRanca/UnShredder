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
rm chap1.aux
rm chap2.aux
rm chap3.aux
rm chap4.aux

evince $file.pdf
