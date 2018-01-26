"""This Python script counts words in a Markdown / LaTeX document(s).

Usage:
	python3 count_words.py <filename or directory>

It will ignore ATX headers, LaTeX & Markdown comments, and LaTeX markup tags.

TODO: inline HTML, Markdown images & tables
"""


def get_filename() -> str:
	"""
	Get the filename from the command line.

	Returns: the first positional argument (the filename)
	"""
	from sys import argv
	try:
		# try to get the filename
		filename = argv[1]
	except IndexError:
		# pass the exception along to the next level
		raise ValueError('no filename position argument!')
	return filename


try:
	# get the filename from the command line
	filename = get_filename()
except ValueError:
	# print the usage information and exit
	print(__doc__)
	from sys import exit
	exit(1)


# specific Markdown files to ignore like READMEs and build artifacts
IGNORED_MD_FILENAMES = ['README.md', 'build.md']

# specific extensions to recognized as Markdown
MARKDOWN_EXTENSIONS = ['.md', 'markdown', '.tex']


def is_md(filename: str) -> bool:
	"""
	Return a boolean determining if the filename is a markdown file.

	Args:
	    filename: the filename to validate

	Returns: true if the filename is a markdown file, false otherwise
	"""
    # iterate over the ignored filenames to ensure the file is valid
	for ignored_file_name in IGNORED_MD_FILENAMES:
		if filename == ignored_file_name:
			return False
    # iterate over the extensions in the accepted extensions
	for markdown_extension in MARKDOWN_EXTENSIONS:
		if markdown_extension == filename[-len(markdown_extension):]:
			return True
	# the filename doesn't have a valid extension, return False
	return False


def markdown_filenames(directory) -> list:
	"""
	Return a list of the filenames in the input directory.

	Args:
		directory: the input directory

	Returns: a list of the markdown files in the input directory.
	"""
	from os import listdir
	try:
		# return a sorted list of the files in the given directory if they
		# are legal markdown files
		return sorted([file for file in listdir(directory) if is_md(file)])
	except FileNotFoundError:
		# catch a file not found error if the directory doesn't exist
		print('{} does not exist!'.format(directory))
		exit(1)


# the sentinel value for LaTeX comment lines
LATEX_COMMENT = '%'

# the sentinel value for Markdown header lines
MARKDOWN_HEADER = '#'


def clean_line(line: str) -> str:
	"""
	Clean a single line and return it.

	Args:
		line: the line of Markdown / LaTeX to clean

	Returns: a cleaned line of text
	"""
	if LATEX_COMMENT in line[:1] or MARKDOWN_HEADER in line[:1]:
		# ignore LaTeX comment lines and Markdown header lines
		return ''
	# strip the line of all whitespace and new lines and append a single space
	return line.rstrip() + ' '


# A regular expression for removing Markdown comments
MARKDOWN_COMMENT_REGEX = r'<!--.+?-->'

# a regular expression for removing LaTeX markup the group in () is the
# optional [] parameters following a markup call.
LATEX_MARKUP_REGEX = r'\\.+?{.+?}(\[.+?\])?'


def clean_contents(contents: str) -> str:
	"""
	Clean and return the contents of a LaTeX / Markdown file.

	Args:
		contents: the contents of the file to clean

	Returns: a file with all markup nonsense removed (just words)
	"""
	clean_contents = contents
	from re import sub
	# remove the markdown comments from the text
	clean_contents = sub(MARKDOWN_COMMENT_REGEX, '', clean_contents)
	# remove the LaTeX markup
	clean_contents = sub(LATEX_MARKUP_REGEX, '', clean_contents)
	# return the clean text
	return clean_contents


def read_file(filename: str) -> str:
	"""
	Read the contents of a single file.

	Args:
		filename: the name of the file to read

	Returns: the string contents of the file
	"""
	if not is_md(filename):
		raise ValueError('filename must have a valid markdown extension')
	# initialize the contents to store from the file
	contents = ''
	# open the file into the contents one line at a time
	with open(filename) as md_file:
		# iterate over each line in the file and write it to the output
		for line in md_file:
			contents += clean_line(line)
	return clean_contents(contents)


def read_dir(directory: str) -> str:
	"""
	Read the contents of every Markdown / LaTeX file in a directory.

	Args:
		directory: the name of the directory to read files from

	Returns: the concatenated contents of the files in the directory
	"""
	# initialize the contents to store from the files
	contents = ''
	# iterate over the files and collect their contents
	for filename in markdown_filenames(directory):
		contents += read_file(f'{directory}/{filename}')
	return contents


def read_contents(filename: str) -> str:
	"""
	Read the contents of a file or directory.

	Args:
		filename: the filename or directory to read

	Returns: the concatenated text from the file(s)
	"""
	from os.path import isdir
	if isdir(filename):
		return read_dir(filename)
	else:
		return read_file(filename)


def words(contents: str) -> int:
	"""
	Return the number of words in a file's contents.

	Args:
		contents: the text to count the words in

	Returns: the number of words in the contents
	"""
	from re import findall
	return findall(r'\w+', contents)


# read the contents of the file
contents = read_contents(filename)
# split the contents into words
words = words(contents)
# count the words in the list
word_count = len(words)
# print the word count to the console
print(f'{word_count} words in {filename}')
