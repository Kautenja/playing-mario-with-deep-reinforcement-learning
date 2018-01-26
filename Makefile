#
# MARK: references
#

# Update the references in the given project directory.
# Args:
# 	$(1): the name of the project directory to update references in
define update_references
# move into the lit review directory and build the master reference file
	cd literature-review && make references
# copy the master reference file to the bib folder of the input directory
	mv literature-review/build/references.bib $(1)/src/bib/references.bib
endef

update_proposal_references:
	$(call update_references,proposal)

update_paper_references:
	$(call update_references,paper)
