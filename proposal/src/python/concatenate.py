"""Concatenate the markdown files into one file."""
from os import listdir


def is_md(filename: str) -> bool:
    """Return a boolean determining if the filename is a markdown file.

    - parameters:
        - filname: the filename to check
    - returns: true if the filename is a markdown file, false otherwise
    """
    if filename == 'README.md' or filename == 'build.md':
        return False
    return '.md' in filename or '.markdown' in filename or '.tex' in filename


def markdown_filenames(directory) -> list:
    """
    Return a list of the filenames in the input directory.

    - parameters:
        - directory: the input directory
    - return: a list of the markdown files in the input directory.
    """
    try:
        return sorted([filename for filename in listdir(directory) if is_md(filename)])
    except FileNotFoundError:
        print('{} does not exist!'.format(directory))
        exit(1)


def concatenate_files(input_directory, output_file_name):
    """Concatenate the input files and write them to the output file.

    - parameters:
        - input_directory: the input directory to find files in
        - output_file_name: the name of the output file
    """
    input_file_names = markdown_filenames(input_directory)
    # open the ouput file to write the concatenation to
    with open(output_file_name, 'w') as outfile:
        # iterate over all the files in the input directory
        for file_name in input_file_names:
            # open the file
            with open('{}/{}'.format(input_directory, file_name)) as md_file:
                # iterate over each line in the file and write it to the output
                for line in md_file:
                    outfile.write(line)
                # add some line breaks between chapters
                outfile.write('\n\n\n')


__all__ = [
    'concatenate_files'
]
