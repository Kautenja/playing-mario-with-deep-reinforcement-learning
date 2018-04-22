# the pointer to the pdflatex binary
PDFLATEX=pdflatex
# the pointer to the bibliography engine (BibTex)
BIBTEX=bibtex

delete_tex_garbage:
	rm -f *.aux *.nav *.log *.out *.snm *.toc *.bbl *.blg *.brf *.swp *.nlo

presentation: delete_tex_garbage
	${PDFLATEX} -halt-on-error presentation | grep -a3 "^!" || true
	@make delete_tex_garbage
