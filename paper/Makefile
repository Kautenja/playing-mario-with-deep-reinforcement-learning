# the linux package manager
APTGET=sudo apt-get
# the macos packager manager
BREW=brew
# the build directory
BUILD=build
# the binary location of `python` the python interpretter for the machine
PYTHON=python3
# the binary location for the pandoc program
PANDOC=pandoc
# the binary location for the pdflatex program
PDFLATEX=pdflatex
# the binary location for the bibtex program
BIBTEX=bibtex

#
# MARK: Default
#

# make everything
default: pdf

#
# MARK: Installation Scripts
#

# do the full installation on a Linux machine using apt-get
linux_install:
	${APTGET} -qq update
	${APTGET} install -y --no-install-recommends texlive-full pandoc pandoc-citeproc

# do the full installation on a MacOS machine using brew
macos_install:
	${BREW} update
	${BREW} cask install mactex
	${BREW} install pandoc
	${BREW} install pandoc-citeproc

# the system this script is executing on
SYSTEM := $(shell uname -s)
# general installation script that checks system
install:
	@echo 'installing for: $(SYSTEM)'
ifeq "$(SYSTEM)" "Darwin"
	make macos_install
endif
ifeq "$(SYSTEM)" "Linux"
	make linux_install
endif

#
# MARK: Markdown Compilation
#

# make the master markdown file
markdown:
	${PYTHON} src/python/compile.py

#
# MARK: LaTeX Compilation
#

# Compile a document using pandoc
# Args:
# 	1: the output file type (i.e. 'tex' or 'pdf')
define pandoc_compile
	${PANDOC} ${BUILD}/${BUILD}.md \
		-o ${BUILD}/${BUILD}.$(1) \
		--template=${BUILD}/template.tex
endef

# generate a latex document
tex: markdown
	$(call pandoc_compile,tex)

#
# MARK: PDF Compilation
#

# generate a pdf document using pdflatex and bibtex on the LaTeX build artifact
# This block looks redundant and dumb, but is the official recipe for compiling
# with Bibtex...
pdf: tex
	cd ${BUILD}; \
	${PDFLATEX} ${BUILD}; \
	${BIBTEX} ${BUILD}; \
	${PDFLATEX} ${BUILD}; \
	${PDFLATEX} ${BUILD}

#
# MARK: Word Counts
#

# count the words in the paper (custom Markdown LaTeX parser)
count_words:
# execute the python script and suppress the output of the command to
# the console
	@${PYTHON} src/python/count_words.py src/md

# Count the words in the paper (pdf).
count_words_pdf: pdf
	@echo 'word count:'
	@ps2ascii ${BUILD}/${BUILD}.pdf | wc -w


