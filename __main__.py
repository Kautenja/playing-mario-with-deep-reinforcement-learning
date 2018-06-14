"""The main execution script for this package."""
# set matplotlib to override default X11 environment
import matplotlib
matplotlib.use('Agg')
# import the main entry point of the application
from src.cli import main
# execute the main entry point of the CLI
main()
