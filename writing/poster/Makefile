ALL: shredderPoster.pdf 

%.pdf: %.tex Makefile
	pdflatex --enable-write18 $< &&	pdflatex $< && pdflatex $<
	evince shredderPoster.pdf 

clean:
	  rm -f *.aux	*.bbl	*.blg	*.log	poster_*.pdf 
