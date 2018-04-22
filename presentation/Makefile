# the pointer to the pdflatex binary
PDFLATEX=pdflatex
# the pointer to the bibliography engine (BibTex)
BIBTEX=bibtex

delete_tex_garbage:
	mkdir -p build
	cd build && rm -f *.aux *.nav *.log *.out *.snm *.toc *.bbl *.blg *.brf *.swp *.nlo

pdf: delete_tex_garbage
	cp presentation.tex build/build.tex
	rsync -r img build
	cd build && ${PDFLATEX} -halt-on-error build | grep -a3 "^!" || true
	@make delete_tex_garbage
