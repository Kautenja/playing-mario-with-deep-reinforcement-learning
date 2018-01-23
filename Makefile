# the pointer to the brew binary for MacOS machines
BREW=brew
# the pointer to the apt-get binary for Linux machines
APT_GET=apt-get


# install dependencies for MacOS platforms
macos_install:
	${BREW} install fceux


# install dependencies for Linux platforms
linux_install:
	${APT_GET} install fceux