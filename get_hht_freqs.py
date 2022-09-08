"""
Performs the Hilbert-Huang transform given a particular file.
"""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import emd
from scipy.fft import rfft, rfftfreq
from numpy.typing import ArrayLike
from typing import Optional

import utils_general as utils_g
from utils_parse import ScoreTimeSeries


def parse_args(args, config):
    parser = utils_g.create_parser(config)

    parser.add_argument(
        'score_ts_dir',
        type=str,
        help=('directory containing the subdirectory "nrc_avg_scores/ "'
              'for the *.csv files containing the, '
              'scores for each window, '
              'with the columns corrresponding to the scores and the rows '
              'to the window number/"time"')
    )

    parser.add_argument(
        'pg_id',
        type=int,
        help='Project Gutenberg ID (no leading zeros) of the book'
    )

    parser.add_argument(
        '--score_ts_subdir', '--subdir',
        type=str,
        default='nrc_avg_scores',
        help=('the subdirectory of score_ts_dir where the files are')
    )

    parser.add_argument(
        '--score_cols',
        nargs='+',
        default=[
            # 'valence', 'arousal', 'dominance',
            # 'goodness', 'energy',
            'power', 'danger', 'structure'
            ],
        type=str,
        help='score columns in score file'
    )

    parser.add_argument(
        '--outputdir',
        default=None,
        type=str,
        help=('output directory for the files')
    )

    parser.add_argument(
        '--min_freq',
        type=str,
        help=('output directory for the files')
    )

    parser.add_argument(
        '--nensembles',
        default=100,
        type=int,
        help=('number of ensembles to generate')
    )

    parser.add_argument(
        '--seed',
        default=60871,
        type=int,
        help=('seed for generating the seed sequence in generating noise for '
              'EEMD')
    )

    parser.add_argument(
        "--resample",
        action="store_true",
        default=True,
        help="uses a sample rate = 1/skip_size for the HHT; if False, sample_rate=1"
    )

    parser.add_argument(
        "--complete",
        action="store_true",
        default=False,
        help="use complete EEMD instead of EEMD"
    )

    parser.add_argument(
        "--interpolate",
        action="store_true",
        default=False,
        help="performs linear interpolation in the case of NaNs"
    )

    parser.add_argument(
        "--standardize", "--stdize",
        action="store_true",
        default=False,
        help="standardizes the time series before taking the IMFs"
    )

    return parser.parse_args(args)


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

    score_ts_dir = Path(args.score_ts_dir)
    subdir = Path(args.score_ts_subdir)
    if args.resample:
        sample_rate = 1.0 / int(args.score_ts_dir.split('_skip=')[1].split('_')[0])
    else:
        sample_rate = 1
    datadir = score_ts_dir / subdir
    pg_id = args.pg_id
    score_cols = args.score_cols
    nensembles = args.nensembles
    standardize = args.standardize

    outputdir = args.outputdir
    if outputdir is None:
        outputdir = score_ts_dir.parent / 'hht'
    if standardize:
        outputdir = f'{outputdir}std'
    outputdir = Path(outputdir)

    outputdir = (Path(f'{outputdir}_nensembles={nensembles}'
                      f'_resample={args.resample}')
                 / score_ts_dir.name)
    utils_g.makedir_if_needed(outputdir)

    sq = np.random.SeedSequence(args.seed)
    seedgen = sq.generate_state(500)        

    sts = ScoreTimeSeries(pg_id, datadir)
    sts.get_raw_ts(set_prop=True, suffix=subdir, interpolate=args.interpolate,
                   stdize=standardize)

    i = 0
    for score_col in score_cols:
        while True:
            try:
                hht_outputdir = outputdir / 'hht' / score_col
                utils_g.makedir_if_needed(hht_outputdir)
                outputfile = hht_outputdir / f'{pg_id}_hht_{score_col}.csv'

                imf_outputdir = outputdir / 'imf' / score_col
                utils_g.makedir_if_needed(imf_outputdir)
                imf_outputfile = imf_outputdir / f'{pg_id}_imf_{score_col}.csv'
                if not args.overwrite:
                    if Path(outputfile).exists():
                        raise Exception(f'Outputfile {outputfile} exists. '
                                        'Set overwrite=True.')

                sts.get_imfs(score_col, seed=seedgen[i],
                             # outputfile=imf_outputfile,
                             outputfile=None,
                             nensembles=nensembles,
                             set_prop=True,
                             complete=args.complete)
                sts.get_hht(score_col, seed=seedgen[i],
                            # outputfile=outputfile,
                            outputfile=None,
                            nensembles=nensembles, sample_rate=sample_rate)
                try:
                    if not standardize:
                        assert np.isclose(pd.DataFrame(sts.imf).mean().sum(),
                                          sts.raw_ts[score_col].mean(),
                                          rtol=0.1)
                    else:
                        assert np.isclose(pd.DataFrame(sts.imf).mean().sum(),
                                          sts.raw_ts[score_col].mean(),
                                          atol=1e-3)

                    logging.info('Writing IMFs...')
                    sts.imf_df.to_csv(imf_outputfile)
                    logging.info(f'Done: see {imf_outputfile}')

                    logging.info('Writing HHTs...')
                    sts.hht_df.to_csv(outputfile)
                    logging.info(f'Done: See {outputfile}')

                    print(pd.DataFrame(sts.imf).mean().sum(),
                                      sts.raw_ts[score_col].mean())

                    break
                except AssertionError:
                    print(pd.DataFrame(sts.imf).mean().sum(),
                                      sts.raw_ts[score_col].mean())
                    logging.warning(f'Imf for seed={seedgen[i]} is incomplete,'
                                    f' shifting to {seedgen[i+1]}')
                    pass

            except IndexError:  # error in emd.sift.ensemble_sift (no IMFs)
                logging.warning(f'Seed {seedgen[i]} did NOT work, shifting to '
                                f'{seedgen[i+1]}')
                pass
            i += 1


if __name__ == '__main__':
    main()
