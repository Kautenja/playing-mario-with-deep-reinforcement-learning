# the name of the terminal node file to use with find
REFERENCE_PATTERN=reference.bib
# the pattern for comment lines and blank lines to ignore. Note that $ are
# operators in Make and thus are escaped. i.e. $$ means $ at the command line.
COMMENT_PATTERN="(%%|^$$|^\s*$$)"
# the name of the build directory
BUILD=build
# the name of the master file to build
REFERENCE_FILE=references.bib

# Make the master file of references.
references:
	mkdir -p ${BUILD}
# 1. cat the files out using find to recursive get all the reference files
# 2. pass the raw text to egrep to filter out comment lines.
# 3. write the single file to disk as the build item
	cat `find . -name ${REFERENCE_PATTERN}` | \
		egrep -v ${COMMENT_PATTERN} > \
		${BUILD}/${REFERENCE_FILE}
