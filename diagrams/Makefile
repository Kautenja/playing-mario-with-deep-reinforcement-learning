# the build folder within this directory
BUILD=build
# the filetype of the output images (this is a PlantUML flag)
OUTPUT_FILETYPE=-tsvg

# make everything in this script
all: clean build uml

# remove any files created by make
clean:
	rm -rf build

# make a build folder for build artifacts
build:
	mkdir -p ${BUILD}

# make the UML diagrams
uml: build *.puml
	plantuml -nbthread auto -o ${BUILD} ${OUTPUT_FILETYPE} *.puml
