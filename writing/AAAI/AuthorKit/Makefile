ALL: poker.pdf 

%.pdf: %.tex Makefile
	pdflatex --enable-write18 $< &&	pdflatex $< && pdflatex $<
	evince poker.pdf 

clean:
	  rm -f *.aux	*.bbl	*.blg	*.log	poster_*.pdf 
