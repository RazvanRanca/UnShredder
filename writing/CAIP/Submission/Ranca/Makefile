ALL: unshredder.pdf 

%.pdf: %.tex Makefile
	pdflatex $< &&	pdflatex $< && pdflatex $<
	evince unshredder.pdf 

clean:
	  rm -f *.aux	*.bbl	*.blg	*.log	unshredder_*.pdf 
