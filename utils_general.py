"""
General utilities
"""
import os
import logging
from datetime import datetime
from operator import attrgetter
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from pathlib import Path
import re
import numpy as np
from glob import glob
import pandas as pd


class SortedMenu(ArgumentDefaultsHelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortedMenu, self).add_arguments(actions)


class CustomFormatter(SortedMenu, RawTextHelpFormatter):
    pass


def create_parser(defaults):
    """
    Creates the parser object with the log arguments included.
    """
    try:
        import cli
        parser = cli.parser()
    except Exception:
        import argparse
        parser = argparse.ArgumentParser(formatter_class=CustomFormatter)

    parser.add_argument(
        "--logdir",
        type=str,
        default=defaults['logdir'],
        help="the log directory"
    )

    parser.add_argument(
        "--logname",
        type=str,
        default=None,
        help="the name of the logfile; specify is using the same logfile in append mode"
    )

    parser.add_argument(
        "--logfile",
        action="store_true",
        default=defaults['logfile'],
        help="create a log file"
    )
    parser.add_argument(
        "--no-logfile",
        dest="logfile",
        action="store_false",
        help="will not log on file"
    )
    parser.add_argument(
        "--logscreen",
        action="store_true",
        default=defaults['logscreen'],
        help="display the log on the screen"
    )
    parser.add_argument(
        "--no-logscreen",
        dest="logscreen",
        action="store_false",
        help="will not log on screen"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=defaults['overwrite'],
        help="overwrites the outputfile if it exists"
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='test run'
    )

    return parser


def makedir_if_needed(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        pass


def log_sysargs(logname, args, script=None, date=None,
                to_file=True, to_screen=False, filemode='a'):
    if date is None:
        date = datetime.now()
    if Path(logname).suffix != '.log':
        log_filename = '{0}_{1}.log'.format(logname,
                                            date.strftime('%Y%m%d-%H%M%S'))
    else:
        log_filename = logname

    if not Path(log_filename).parent.exists():
        makedir_if_needed(Path(log_filename).parent)

    if not os.path.exists(log_filename):
        filemode = 'w'

    handlers = []
    if to_file:
        fh = logging.FileHandler(log_filename, mode=filemode)  # new log everytime
        formatter = logging.Formatter(
                fmt='%(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
            )
        fh.setFormatter(formatter)
        handlers.append(fh)

    if to_screen:
        sh = logging.StreamHandler()
        formatter = logging.Formatter(
                fmt='%(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
            )
        sh.setFormatter(formatter)
        handlers.append(sh)

    if len(handlers) != 0:
        logging.basicConfig(
                # filename=log_filename,
                # filemode='w', # new log everytime
                # format='%(asctime)s %(message)s',
                # datefmt='%m/%d/%Y %I:%M:%S %p',
                level=logging.INFO,
                handlers=handlers,
            )

        if script is not None:
            info_text = '\n\n******SCRIPT PARAMETERS******\n\n'

            info_text += script + '\n'

            for k, v in vars(args).items():
                info_text += str(k) + '\t' + str(v) + '\n'
            info_text += '\n*****************************\n'

            logging.info(info_text)


def walk(path):
    """
    Get all files in subdirectories
    """
    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p.resolve()


def get_all_files(path):
    for p in walk(Path(path)):
        yield p


def glob_regex(pattern, path, sort=True):
    path = str(path)
    files = [f for f in os.listdir(path) if re.search(pattern, f)]
    if sort:
        files = sorted(files)
    return files


def get_files_in_dir(directory, test=False, seed=1234):
    directory = Path(directory)
    filelist = sorted(list(directory.iterdir()))
    if test:
        rng = np.random.default_rng(seed=seed)
        filelist = rng.choice(filelist, size=20, replace=False)
        filelist = filelist.tolist()

    return filelist


def get_pgids_in_dir(targetdir, ext='csv', suffix=None, as_frame=False):
    if suffix is None:
        search_term = str(targetdir / f'*.{ext}')
    else:
        search_term = str(targetdir / f'*{suffix}*.{ext}')

    pg_ids = [int(Path(i).stem.split('_')[0]) for i in glob(search_term)]

    srs = pd.Series(pg_ids).rename('pg_ids')

    if as_frame:
        srs = srs.to_frame()

    return srs


def rename_dict_keys(d, key_mapping):
    for orig, new in key_mapping.items():
        if orig in d.keys():
            d[new] = d.pop(orig)

    return d


def try_gz_suffix(filename, as_path=True):
    fname = Path(filename)
    if not fname.exists():
        fname = Path(f'{fname}.gz')

    try:
        assert fname.exists()
    except AssertionError:
        raise AssertionError(f'File does not exist: {fname}')

    if as_path:
        fname = Path(fname)
    else:
        fname = str(fname)

    return fname


def try_no_gz_suffix(filename, as_path=True):
    fname = Path(filename)
    if not fname.exists():
        fname = Path(str(fname).replace('.gz', ''))

    try:
        assert fname.exists()
    except AssertionError:
        raise AssertionError(f'File does not exist: {fname}')

    if as_path:
        fname = Path(fname)
    else:
        fname = str(fname)

    return fname


def get_seed_index(seed, srs, label=None):
    """
    Gets the row number of the seed in a seed file read as a pandas Series.
    """
    idx = srs[srs == seed].index
    if label is not None:
        label = 'Index'
    else:
        label = f'{label} index'
    try:
        assert len(idx) == 1
    except AssertionError:
        raise Exception(f'{label} invalid: {idx}')

    return idx


class NestedFunctions(object):
    """
    Functions with kwargs that are recast as functions of functions.
    Useful when used with pd.DataFrame.agg.
    """
    @staticmethod
    def percentile_fxn(percentile):
        def pctile(x):
            return np.percentile(x, percentile, axis=0)

        pctile.__name__ = f'pctile_{percentile}'

        return pctile


def split_arr_into_chunks(arr, chunksize):
    """
    Splits an array into chunks of length chunksize; the last chunk
    will be the remainder.
    """
    return np.split(arr, np.arange(chunksize, len(arr), chunksize))


class BootstrapGroupby(object):
    """Functions necessary for bootstrapping."""
    @classmethod
    def subsample_df(cls, df, grp_by='pg_id', n=40, seed=42):
        """
        Groups the dataframe by `grp_by` and subsamples, without replacement,
        `n` entries. In the case of rows corresponding to different seeds, this
        ensures that we have the same number of seeds for all `pg_ids`.
        """
        size_srs = df.groupby(grp_by).size()
        assert df.index.name == grp_by
        idx_pass = size_srs[size_srs >= n].index

        try:
            assert len(idx_pass) > 0
        except AssertionError:
            raise AssertionError(f'Not enough for subsampling: n={n}\n{df.head}')
        # try:
        #     assert size_srs.min() >= n
        # except AssertionError:
        #     msg = f'Minimum n in series is {size_srs.min()}. Replace n to proceed.'
        #     raise AssertionError(msg)

        subsample_df = df.loc[idx_pass].groupby(grp_by).apply(
            lambda grp: grp.sample(n=n, replace=False, random_state=seed)
            ).droplevel(1)

        return subsample_df

    @classmethod
    def generate_bootstrap_stats(cls, df, B, statistics,
                                 grp_by='pg_id',
                                 fillna=0, seed=None):
        """
        Can be applied to a dataframe or to a groupby object. We set fillna=0
        by default since we're working with NaN in variances of IMFs (an IMF
        that doesn't exist will have an energy of 0)

        Parameters
        ----------
        statistics: array-like
            list of functions that can be fed into pd.DataFrame.agg
        """
        df = df.fillna(fillna)  # copy
        sq = np.random.SeedSequence(seed)
        seedgen = sq.generate_state(B)
        stat_res = [None] * B
        for boot in range(B):
            rng = np.random.default_rng(seedgen[boot])
            boot_df = pd.DataFrame(
                df.values[rng.choice(df.shape[0], size=df.shape[0], replace=True)],
                columns=df.columns,
                index=df.index  # should all be the same, since this is for a particular pg_id
                )
            # takes too long
            # boot_df = df.sample(frac=1, replace=True, random_state=seedgen[boot])
            res = boot_df.agg(statistics)
            res.index.name = 'stat'
            res['boot_idx'] = boot
            stat_res[boot] = res
        stat_df = pd.concat(stat_res, axis=0).set_index('boot_idx', append=True).unstack(level=0)
        stat_df = stat_df.agg([
            np.mean,
            np.median,
            NestedFunctions.percentile_fxn(99.5),
            NestedFunctions.percentile_fxn(97.5),
            NestedFunctions.percentile_fxn(2.5),
            NestedFunctions.percentile_fxn(0.5),
        ])

        return stat_df


def get_seeds(seed_file=None, start_from_seed=None, end_at_seed=None,
              start_idx=None, end_idx=None, seeds=None):
    if seed_file is None and seeds is None:
        seeds = None

    elif seed_file is not None:
        seed_file = Path(seed_file)
        seeds = pd.read_csv(seed_file, header=None,
                            squeeze=True)
    else:
        seeds = seeds

    if seeds is not None:
        if start_from_seed is not None:
            assert start_idx is not None
            start_idx = get_seed_index(start_from_seed, seeds)

        if end_at_seed is not None:
            assert end_idx is not None
            end_idx = get_seed_index(end_at_seed, seeds)

        if start_idx is not None and end_idx is not None:
            seeds = seeds[start_idx:end_idx + 1]
        elif start_idx is not None and end_idx is None:
            seeds = seeds[start_idx]
        elif start_idx is None and end_idx is not None:
            seeds = seeds[:end_idx+1]
        else:
            pass

    return seeds


def nullable_string(val: str):
    if val.lower() in ['none', '']:
        return None
    else:
        return val


class sdict(dict):
    """
    Creates a dictionary where the keys are strings, an which accepts
    float and int keys, since it automatically converts them to strings.

    This ensures that the dictionary keys are not affected by how
    Python treats floats.
    """
    def __init__(self, *args, **kw):
        super(sdict,self).__init__(*args, **kw)
        # self.itemlist = super(odict,self).keys()

    def __getitem__(self, key):
        if not isinstance(key, str):
            key = str(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            key = str(key)
        return super().__setitem__(key, value)


def get_window_skip_size_from_str(lbl):
    N_w = lbl.split('window=')[-1].split('_')[0]
    N_s = lbl.split('skip=')[-1].split('_')[0]

    try:
        N_w = int(N_w)
    except TypeError:
        pass

    try:
        N_s = int(N_s)
    except TypeError:
        pass

    return N_w, N_s
