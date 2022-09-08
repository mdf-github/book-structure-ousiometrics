"""
Writes files for the word counts (total and NRC lexicon) and scores for
a particular text file given sliding window parameters. If sliding window
parameters are not given, the full text is considered in its entirety
as a single window.
"""
import sys
import logging
from pathlib import Path

import utils_parse as utils_p
import utils_general as utils_g


def parse_args(args, config):
    parser = utils_g.create_parser(config)

    parser.add_argument(
        'bookfile',
        type=str,
        help='path for the *_text.txt (clean) files'
    )

    parser.add_argument(
        'outputdir',
        type=str,
        help='directory to place the folder containing the output files'
    )

    parser.add_argument(
        '--prefix',
        default=None,
        type=str,
        help='prefix used for the output file; if not given, uses the filename'
    )

    parser.add_argument(
        '--N_w',
        default=None,
        type=int,
        help='size of sliding window; if not given, a single window of the length of the piece is considered'
    )

    parser.add_argument(
        '--n',
        default=None,
        type=int,
        help='expected number of data points required; if not given, the text is considered in its entirety'
    )

    parser.add_argument(
        '--N_s',
        default=None,
        type=int,
        help='expected number of skip points required; if not given, the text is considered in its entirety'
    )

    parser.add_argument(
        '--thresh',
        default=0.7,
        type=str,
        help='threshold to include the last window if it is >=thresh * N_w'
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="shuffle the text"
    )

    parser.add_argument(
        "--shuffle_sentences",
        action="store_true",
        default=False,
        help="shuffle the sentences before tokenizing"
    )

    # parser.add_argument(
    #     '--metadata',
    #     default=None,
    #     type=str,
    #     help='the path for the metadata file; if None, uses the hard-coded default path'
    # )

    parser.add_argument(
        '--scorefile',
        default=None,
        type=str,
        help='the path for the score file; if None, uses the hard-coded default path'
    )

    parser.add_argument(
        '--score_cols',
        nargs='+',
        default=[
            'valence', 'arousal', 'dominance',
            'goodness', 'energy',
            'power', 'danger', 'structure'],
        type=str,
        help='score columns in score file'
    )

    parser.add_argument(
        '--remove_words',
        nargs='*',
        default=None,
        type=str,
        help='words to be removed from score file'
    )

    parser.add_argument(
        '--seed',
        default=1234,
        # type=int,
        help='seed for shuffling text'
    )

    parser.add_argument(
        "--gzip",
        action="store_true",
        default=True,
        help="shuffle the sentences before tokenizing"
    )

    return parser.parse_args(args)


def write_texttrends_output(texttrends_obj, outputdir_base, fileprefix,
                            overwrite=False, gzip=True):
    utils_g.makedir_if_needed(outputdir_base)

    output_types = ['word_counts', 'nrc_counts', 'nrc_avg_scores']

    for output in output_types:
        outputdir = outputdir_base / output
        utils_g.makedir_if_needed(outputdir)

        outputfile = outputdir / f'{fileprefix}_{output}.csv'

        kwargs = dict()
        if gzip:
            outputfile = Path(f'{outputfile}.gz')
            kwargs['compression'] = 'gzip'

        if not overwrite:
            if outputfile.exists():
                raise RuntimeError(f'File {outputfile} exists. Set overwrite=True.')
        getattr(texttrends_obj, output).to_csv(outputfile, **kwargs)
        logging.info(f'Done: {outputfile}')


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
    logname = Path(config['logdir']) / f'{Path(sys.argv[0]).stem}'

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

    bookfile = args.bookfile
    fileprefix = args.prefix
    if fileprefix is None:
        fileprefix = Path(bookfile).stem.split('_text')[0][2:]
    tokenizer_kw = None
    preprocessor_kw = None

    scorefile = args.scorefile
    if scorefile is None:
        scorefile = Path.cwd() / 'ousiometer_scores' / 'ousiometer_verb_plurals.tsv'
    score_cols = args.score_cols
    remove_words = args.remove_words

    scores = utils_p.get_raw_scores(scorefile,
                                    score_cols=score_cols,
                                    remove_words=remove_words)

    with open(bookfile, 'r') as f:
        txt = f.read()

    N_w = args.N_w
    n = args.n
    N_s = args.N_s
    seed = args.seed
    try:
        seed = int(seed)
    except ValueError:
        seed = None

    if N_w is not None:
        thresh = float(args.thresh)
    else:
        thresh = None
    shuffle = args.shuffle
    shuffle_sentences = args.shuffle_sentences
    if shuffle_sentences:
        shuffle = False  # we do NOT do a reshuffle after shuffling sentences

    logging.info('Generating sliding window...')
    if N_w is None:
        text_list = [txt]  # list of a single element to mirror a sliding window
    else:
        text_list = utils_p.SlidingWindowText(
                        txt, n=n, N_w=N_w, N_s=N_s, thresh=thresh,
                        random=shuffle, seed=seed,
                        shuffle_sentences=shuffle_sentences).sliding_windows

    logging.info('Done.')
    logging.info('Generating TextTrends object...')
    texttrends_obj = utils_p.TextTrends(text_list, scores,
                                        tokenizer_kw=tokenizer_kw,
                                        preprocessor_kw=preprocessor_kw)
    logging.info('Done.')

    outputdir_subdir = f'window={N_w}_n={n}_skip={N_s}_thresh={thresh}_shuffle={shuffle}'
    if shuffle:
        outputdir_subdir = f'{outputdir_subdir}_seed={seed}'
    if shuffle_sentences:
        outputdir_subdir = f'{outputdir_subdir}_seed={seed}_shufflesentences={shuffle_sentences}'

    outputdir_base = Path(args.outputdir) / outputdir_subdir
    write_texttrends_output(texttrends_obj,
                            outputdir_base=outputdir_base,
                            fileprefix=fileprefix,
                            overwrite=args.overwrite,
                            gzip=args.gzip)

    logging.info('DONE.')


if __name__ == '__main__':
    main()
