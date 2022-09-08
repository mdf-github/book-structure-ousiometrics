"""
Removes the headers and footers from the Gutenberg raw files and creates
clean versions of the text.

Code copied/adapted from the standardized Gutenberg corpus github,
https://github.com/pgcorpus/gutenberg
"""

from __future__ import unicode_literals
import os
import sys
import logging
import gzip
from pathlib import Path
import utils_general as utils_g


def parse_args(args, config):
    parser = utils_g.create_parser(config)

    parser.add_argument(
        'src_dir',
        type=str,
        help='source data directory containing *.txt* files in *any* subdirectory'
    )

    parser.add_argument(
        'target_dir',
        type=str,
        help='target directory to write all clean files'
    )

    parser.add_argument(
        '--suffix',
        default='.txt',
        type=str,
        help='file suffix/suffix header for the target files')

    return parser.parse_args(args)


def cleanup(path, text_dir, suffix='.txt', overwrite=False):
    """
    Remove headers from raw txt file.

    Parameters
    ----------
    path : string
        Path to the PG****_raw.txt file
    text_dir: target directory
    """
    # PG_number = path.split("/")[-1].split("_")[0][2:]
    if not isinstance(path, Path):
        path = Path(path)
    PG_number = path.name.split(suffix)[0][2:]

    source_file = os.path.join(text_dir, "PG%s_text.txt" % PG_number)
    if not overwrite:
        if Path(source_file).exists():
            logging.warning(f'File {source_file} exists. Set overwrite=True.')
            return

    if path.suffix == '.gzip':
        with gzip.open(path, 'rt') as f:
            text = f.read()
    else:
        with open(path) as f:
            text = f.read()

    clean = strip_headers(text)
    with open(source_file, "w") as f:
        f.write(clean)


############

TEXT_START_MARKERS = frozenset((
    "*END*THE SMALL PRINT",
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "This etext was prepared by",
    "E-text prepared by",
    "Produced by",
    "Distributed Proofreading Team",
    "Proofreading Team at http://www.pgdp.net",
    "http://gallica.bnf.fr)",
    "      http://archive.org/details/",
    "http://www.pgdp.net",
    "by The Internet Archive)",
    "by The Internet Archive/Canadian Libraries",
    "by The Internet Archive/American Libraries",
    "public domain material from the Internet Archive",
    "Internet Archive)",
    "Internet Archive/Canadian Libraries",
    "Internet Archive/American Libraries",
    "material from the Google Print project",
    "*END THE SMALL PRINT",
    "***START OF THE PROJECT GUTENBERG",
    "This etext was produced by",
    "*** START OF THE COPYRIGHTED",
    "The Project Gutenberg",
    "http://gutenberg.spiegel.de/ erreichbar.",
    "Project Runeberg publishes",
    "Beginning of this Project Gutenberg",
    "Project Gutenberg Online Distributed",
    "Gutenberg Online Distributed",
    "the Project Gutenberg Online Distributed",
    "Project Gutenberg TEI",
    "This eBook was prepared by",
    "http://gutenberg2000.de erreichbar.",
    "This Etext was prepared by",
    "This Project Gutenberg Etext was prepared by",
    "Gutenberg Distributed Proofreaders",
    "Project Gutenberg Distributed Proofreaders",
    "the Project Gutenberg Online Distributed Proofreading Team",
    "**The Project Gutenberg",
    "*SMALL PRINT!",
    "More information about this book is at the top of this file.",
    "tells you about restrictions in how the file may be used.",
    "l'authorization à les utilizer pour preparer ce texte.",
    "of the etext through OCR.",
    "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
    "We need your donations more than ever!",
    " *** START OF THIS PROJECT GUTENBERG",
    "****     SMALL PRINT!",
    '["Small Print" V.',
    '      (http://www.ibiblio.org/gutenberg/',
    'and the Project Gutenberg Online Distributed Proofreading Team',
    'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
    '                this Project Gutenberg edition.',
))


TEXT_END_MARKERS = frozenset((
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of The Project Gutenberg",
    "Ende dieses Project Gutenberg",
    "by Project Gutenberg",
    "End of Project Gutenberg",
    "End of this Project Gutenberg",
    "Ende dieses Projekt Gutenberg",
    "        ***END OF THE PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED",
    "End of this is COPYRIGHTED",
    "Ende dieses Etextes ",
    "Ende dieses Project Gutenber",
    "Ende diese Project Gutenberg",
    "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
    "Fin de Project Gutenberg",
    "The Project Gutenberg Etext of ",
    "Ce document fut presente en lecture",
    "Ce document fut présenté en lecture",
    "More information about this book is at the top of this file.",
    "We need your donations more than ever!",
    "END OF PROJECT GUTENBERG",
    " End of the Project Gutenberg",
    " *** END OF THIS PROJECT GUTENBERG",
))


LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))
LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))


def strip_headers(text):
    """
    Remove lines that are part of the Project Gutenberg header or footer.

    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp

    Args:
        text (unicode): The body of the text to clean up.

    Returns:
        unicode: The text with any non-text content removed.

    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    config = {
        'logdir': Path.cwd() / 'logs',
        'logfile': False,
        'logscreen': True,
        'overwrite': False,
    }

    args = parse_args(args, config)

    utils_g.makedir_if_needed(config['logdir'])

    if args.logname is None:
        logname = Path(config['logdir']) / f'{Path(sys.argv[0]).stem}'
    else:
        logname = Path(config['logdir']) / args.logname

    utils_g.log_sysargs(logname, args, script=sys.argv[0],
                        to_file=args.logfile,
                        to_screen=args.logscreen
                        )

    logging.info('Initialized logging...')
    logging.info(args)

    src_dir = Path(args.src_dir)
    target_dir = Path(args.target_dir)
    suffix = args.suffix

    utils_g.makedir_if_needed(target_dir)

    srcfiles = [f for f in utils_g.get_all_files(src_dir) if suffix in f.name]

    for i, path in enumerate(srcfiles):
        logging.info(f'Cleaning up {path}')
        cleanup(path, target_dir, overwrite=args.overwrite)

    logging.info('DONE.')


if __name__ == '__main__':
    main()
