# the build directory
BUILD=build

#
# MARK: build artifacts
#

# make the build folder
build_folder:
	mkdir -p ${BUILD}

#
# MARK: installation
#

# install dependencies for the project
install:
	cd proposal && make install
	cd paper && make install
	python3 -m pip install -r requirements.txt

#
# MARK: references
#

# Update the references in the given project directory.
# Args:
# 	$(1): the name of the project directory to update references in
define update_references
	cd literature-review && make references
	mv literature-review/build/references.bib $(1)/src/bib/references.bib
endef

# update the bibliography for the proposal
update_proposal_references:
	$(call update_references,proposal)

# update the bibliography for the paper
update_paper_references:
	$(call update_references,paper)

# Move into the given directory, build its paper, and copy it into this
# directory's build directory.
# Args:
#   $(1): the directory of the
define make_pdf
	cd $(1) && make pdf
	cp $(1)/build/build.pdf ${BUILD}/$(1).pdf
endef

# make the proposal
proposal: build_folder
	$(call make_pdf,proposal)

# make the paper
paper: build_folder
	$(call make_pdf,paper)

#
# MARK: Python code
#

# run the unittests
test:
	python3 -m unittest discover src
