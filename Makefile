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

update_proposal_references:
	$(call update_references,proposal)

update_paper_references:
	$(call update_references,paper)
