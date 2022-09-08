"""
Utilities for parsing
"""
import pandas as pd
import numpy as np
import os
import logging
from glob import glob
from pathlib import Path
import nltk
# from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
import itertools
import string
from typing import Optional, Union
from numpy.typing import ArrayLike
from copy import copy
from collections import OrderedDict
import emd
from scipy.sparse import csr_matrix

import utils_general as utils_g


def get_raw_scores(scorefile: Union[str, Path],
                   score_cols: ArrayLike,
                   remove_words: Optional[ArrayLike] = None):
    scorefile = Path(scorefile)
    scores = pd.read_csv(scorefile, sep='\t', index_col=0, na_filter=False)
    try:
        scores = scores[score_cols]
    except KeyError:
        pass
    if remove_words is not None:
        scores.drop(index=remove_words, inplace=True)

    return scores


class Preprocessor(object):
    def __call__(self, text: str):
        text = text.lower()
        # split up contractions
        text = re.sub(r"’", r"'", text)
        text = re.sub(r"(\w)*(\d)+(\w)*", r" ", text)
        text = re.sub(r"n't", r" not", text)
        text = re.sub(r"'s", r" ", text)  # either "is" or possessive so omit
        text = re.sub(r"'d", r" ", text)  # either "had" or "would", but always precedes a more important verb
        text = re.sub(r"'m", r" ", text)  # because "'s" is set to " "
        text = re.sub(r"'re", r" ", text)
        text = re.sub(r"'ve", r" have", text)
        text = re.sub(r"'ll", r" will", text)

        return text


class Tokenizer(object):
    """Tokenizes into strings with no non-word and digit characters."""
    def __init__(self, stemmed=False, lemmatized=False):
        self.stemmed = stemmed
        self.lemmatized = lemmatized

    def __call__(self, documents: str):
        if self.stemmed:
            stemmer = PorterStemmer()
            fxn = stemmer.stem
        elif self.lemmatized:
            lemmatizer = WordNetLemmatizer()
            fxn = lemmatizer.lemmatize
        else:
            fxn = lambda x: x
        return [
            fxn(term)
            for term in Tokenizer.custom_tokenizer(documents)
            if term.isalpha()
        ]

    @staticmethod
    def custom_tokenizer(text: str):
        """
        Splits `text` by whitespace and removes the non-word characters
        and digits from each element of the resulting list.
        """
        tokenizer = nltk.RegexpTokenizer(r"[^\W\d_]+")
        return tokenizer.tokenize(text)


def get_word_counts(sentences=None, tokenizer_kw=None, preprocessor_kw=None):
    if sentences is None:
        # test
        sentences = ["12 It does make a person think and w-wonder. Don't you think?", "12 It does make a person think and w-w-wonder. You thought and wondered?" ]

    if tokenizer_kw is None:
        tokenizer_kw = {}
    if preprocessor_kw is None:
        preprocessor_kw = {}
    vect = CountVectorizer(
        tokenizer=Tokenizer(**tokenizer_kw),
        preprocessor=Preprocessor(**preprocessor_kw)
    )

    try:
        X = vect.fit_transform(sentences)
        word_df = pd.DataFrame(data=X.toarray())
        idx_to_word = {v: k for k, v in vect.vocabulary_.items()}
        word_df = word_df.rename(columns=idx_to_word).transpose()
    except ValueError:  # when sentence is an empty string
        word_df = None

    logging.info('Done: Word counts obtained.')

    return word_df


class TextTrends(object):
    """
    Obtains the word counts and average scores for a list of sentences
    `text_list`.
    """
    def __init__(self, text_list: ArrayLike, scores: pd.DataFrame,
                 tokenizer_kw: Optional[dict] = None,
                 preprocessor_kw: Optional[dict] = None):
        text_list, text_list_copy = itertools.tee(text_list, 2)
        self.text_list = text_list
        self.tokenizer_kw = tokenizer_kw
        self.preprocessor_kw = preprocessor_kw

        self.word_counts = self.set_word_counts(text_list_copy)
        self.nrc_counts = self.set_nrc_counts(scores)
        self.nrc_avg_scores = self.set_nrc_avg_scores(scores)

    def set_word_counts(self, text_list):
        return get_word_counts(
            sentences=text_list,
            tokenizer_kw=self.tokenizer_kw,
            preprocessor_kw=self.preprocessor_kw
        )

    def set_nrc_counts(self, scores):
        if self.word_counts is None:
            nrc_counts = None
        else:
            nrc_counts = self.word_counts.reindex(
                scores.index).fillna(0).astype('int')

        return nrc_counts

    def set_nrc_avg_scores(self, scores):
        if self.word_counts is None:
            return None

        def _div_or_nan(x, y):
            if y != 0:
                return x / y
            else:
                return np.nan

        nrc_avg_scores = {}
        for colname in scores.columns:
            logging.info(f'Computing average scores for {colname}...')
            # _weighted_scores = self.nrc_counts.apply(lambda col: col.to_numpy() * scores[colname].to_numpy())
            # replace with equivalent below
            # _weighted_scores = pd.DataFrame(np.multiply(scores[colname].to_numpy(), self.nrc_counts.to_numpy().T).T,
            #                                 index=self.nrc_counts.index, columns=self.nrc_counts.columns)
            # replace with something even better
            x = csr_matrix(self.nrc_counts)
            y = csr_matrix(scores[colname]).T
            _weighted_scores = pd.DataFrame.sparse.from_spmatrix(
                x.multiply(y),
                index=self.nrc_counts.index,
                columns=self.nrc_counts.columns)
            logging.info('Weighted scores...')
            _nonzero_weighted_scores = _weighted_scores.loc[(_weighted_scores != 0).any(axis=1)]
            logging.info('Nonzero weighted scores...')
            try:
                _df = _nonzero_weighted_scores.sum(axis=0) / self.nrc_counts.sum(axis=0)
            except ZeroDivisionError:
                srs_a = _nonzero_weighted_scores.sum(axis=0)
                srs_b = self.nrc_counts.sum(axis=0)
                dummy = pd.DataFrame({'a': srs_a, 'b': srs_b})

                dummy['new'] = dummy.apply(lambda row: _div_or_nan(row['a'], row['b']), axis=1)
                _df = dummy['new'].copy()
                del dummy, srs_a, srs_b
            nrc_avg_scores[colname] = _df
            logging.info(f'Done for {colname}')

        return pd.DataFrame(nrc_avg_scores, dtype='float64')

    def evaluate_text_list(self):
        """
        Evaluates the generator)
        """
        self.text_list = [n for n in self.text_list]


def window(seq, size=3, step=1, thresh=1):
    """
    Creates an iterator of a sliding window of size `size`
    that skips every `step` number of steps. The final
    entry of the iterator should be ast least of length
    `thresh*size`.
    """
    if size < 1 or step < 1:
        raise ValueError("Nobody likes infinite loops.")
    it = iter(seq)
    result = list(itertools.islice(it, size))
    while len(result) <= size and len(result) >= thresh * size:
        yield result
        if step >= size:
            result = list(itertools.islice(it, step-size, step))
        else:
            result = result[step:] + list(itertools.islice(it, step))


def clean_text(sentence):
    sentence = re.sub('Chapter', '', sentence) ## removes the "Chapter" headings
    sentence = re.sub(r'(\d+)', '', sentence) ## removes the chapter numbers
    sentence = re.sub(r"[^\w’']", ' ', sentence) ## removes punctuation excluding the ones used for contractions (’')
    sentence = re.sub(r'(\s+)', ' ', sentence) ## converts all whitespace to a single space
    return sentence


def get_list(text):
    full_text = text.strip()
    pp = Preprocessor()
    return Tokenizer.custom_tokenizer(pp(full_text))


class SlidingWindowText(object):
    def __init__(self, full_text,
                 n=1000, N_w=10000, thresh=0.7, N_s=None,
                 random=False, seed=1234,
                 shuffle_sentences=False,
                 tokenizer_kw=None, preprocessor_kw=None):
        """
        Full text is the entire text of the book, minus all the opening credits. Ideally it starts after ``Chapter 1\n''
        """
        self.full_text = full_text
        self.word_list = self.set_word_list(
            random=random, seed=seed,
            shuffle_sentences=shuffle_sentences,
            tokenizer_kw=tokenizer_kw, preprocessor_kw=preprocessor_kw)
        self.thresh = thresh
        self.set_window_params(N_w, n, N_s)

    def randomize_sentences(self, text, seed: Optional[int] = None):
        """
        Shuffles at the sentence level.
        """
        tokenizer = nltk.RegexpTokenizer(r"[?!]+|(?<!\.)\.{1,2}(?!\.)", gaps=True)
        tokenized_punc = tokenizer.tokenize(text)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(tokenized_punc)

        return ' '.join(tokenized_punc)

    def set_word_list(self, random: bool, seed: Optional[int] = None,
                      tokenizer_kw=None, preprocessor_kw=None,
                      shuffle_sentences=False):
        """
        Generates the full text as a list of parsed tokens.
        """
        if tokenizer_kw is None:
            tokenizer_kw = {}
        if preprocessor_kw is None:
            preprocessor_kw = {}

        tokenizer = Tokenizer(**tokenizer_kw)
        pp = Preprocessor(**preprocessor_kw)
        if not shuffle_sentences:
            full_text = self.full_text
            logging.info('Tokenizing...')
            word_list = tokenizer(pp(full_text))
            logging.info('Done.')
            if random:
                logging.info('Randomizing word list...')
                rng = np.random.default_rng(seed=seed)
                rng.shuffle(word_list)
                logging.info('Done.')
        else:
            logging.info('Randomizing sentences...')
            full_text = self.randomize_sentences(self.full_text, seed=seed)
            logging.info('Done.')
            logging.info('Tokenizing...')
            word_list = tokenizer(pp(full_text))
            logging.info('Done.')

        return word_list

    def set_window_params(self, N_w, n, N_s):
        N = len(self.word_list)
        self.N_w = N_w
        if N_w is None:
            self.n = None
            self.N_s = None
        else:
            assert not all([n is None, N_s is None])
            if n is None:
                self.N_s = N_s
                self.n = round((N - (N_w + 1)) / N_s)
            elif N_s is None:
                self.n = n
                self.N_s = round((N - (N_w + 1)) / n)
    # @property
    # def N_w(self):
    #     return self.__N_w

    # @N_w.setter
    # def N_w(self, val):
    #     self.__N_w = val

    # @property
    # def n(self):
    #     return self.__n

    # @n.setter
    # def n(self, val):
    #     self.__n = val

    # @property
    # def N_s(self):
    #     N_w = self.N_w
    #     n = self.n
    #     N = len(self.word_list)

    #     return round((N - (N_w + 1)) / n)  # skipping size

    @property
    def sliding_windows(self):
        """
        Generates a list of "sentences", with each "sentence" being a window
        of length N_w.
        """
        sliding_iterator = window(self.word_list,
                                  size=self.N_w,
                                  step=self.N_s,
                                  thresh=self.thresh)

        def _combine_words(word_list):
            return ' '.join(word_list)

        sliding_iterator = map(_combine_words, sliding_iterator)

        return sliding_iterator


class FullText(object):
    def __init__(self, srcfiles, tokenizer_kw=None, preprocessor_kw=None):
        self.srcfiles = srcfiles
        self.book_text = self.set_book_text(srcfiles,
                                            tokenizer_kw=tokenizer_kw,
                                            preprocessor_kw=preprocessor_kw)

    def set_book_text(self, srcfiles, tokenizer_kw=None, preprocessor_kw=None):
        if tokenizer_kw is None:
            tokenizer_kw = {}
        if preprocessor_kw is None:
            preprocessor_kw = {}
        _book_text_list = [None] * len(srcfiles)
        for numbook, bookfile in enumerate(srcfiles):
            with open(bookfile, 'r') as f:
                txt = f.read()
            pp = Preprocessor(**preprocessor_kw)
            tknz = Tokenizer(**tokenizer_kw)
            _book_text_list[numbook] = (numbook, tknz(pp(txt.split('Chapter 1\n')[1])))

        book_text = OrderedDict(_book_text_list)

        return book_text


def get_text_portions(txt, anchor='snape', threshold=10000):
    """
    Returns the word indices for the occurrence of the anchor word for each partition based on the interevent time threshold.
    """
    txt_srs = pd.Series(txt)
    iet = txt_srs[txt_srs == anchor].rename_axis('word_idx').reset_index()['word_idx'].diff().rename('diff')

    iet_word_idx_df = pd.concat([iet.to_frame(), pd.DataFrame({'word_idx': txt_srs[txt_srs == anchor].index})], axis=1)

    # get the word index of the breakpoints, as well as the corresponding IETs
    breakpoints = iet_word_idx_df[iet_word_idx_df['diff'] > threshold]

    # get the index in the iet_word_idx_df for the corresponding breakpoints
    anchor_breakpoints = iet_word_idx_df.reindex(breakpoints.index)['word_idx'].rename_axis('idx').reset_index()
    # the index is then used to get the range of words within each burst of anchor words with a given IET threshold
    partition = anchor_breakpoints.apply(
        lambda row: iet_word_idx_df.loc[:row['idx'] - 1, 'word_idx'].tolist() if row.name == 0 else iet_word_idx_df.loc[anchor_breakpoints.loc[row.name - 1, 'idx']:row['idx'] - 1, 'word_idx'].tolist(),
        axis=1)

    return partition


def get_maintext_lines_gutenberg(raw_text):
    """
    from Andy Reagan's repo
    https://github.com/andyreagan/core-stories/blob/master/src/bookclass.py#L39
    """
    lines = raw_text.split("\n")
    start_book_i = 0
    end_book_i = len(lines)-1
    # pass 1, this format is easy and gets 78.9% of books
    start1 = "START OF THIS PROJECT GUTENBERG EBOOK"
    start2 = "START OF THE PROJECT GUTENBERG EBOOK"
    end1 = "END OF THIS PROJECT GUTENBERG EBOOK"
    end2 = "END OF THE PROJECT GUTENBERG EBOOK"
    end3 = "END OF PROJECT GUTENBERG"
    for j, line in enumerate(lines):
        if (start1 in line) or (start2 in line):
            # and "***" in line and start_book[i] == 0 and j<.25*len(lines):
            start_book_i = j
        end_in_line = end1 in line or end2 in line or end3 in line.upper()
        if end_in_line and (end_book_i == (len(lines)-1)):
            #  and "***" in line and j>.75*len(lines)
            end_book_i = j
    # pass 2, this will bring us to 99%
    if (start_book_i == 0) and (end_book_i == len(lines)-1):
        for j, line in enumerate(lines):
            if ("end" in line.lower() or "****" in line) and  "small print" in line.lower() and j<.5*len(lines):
                start_book_i = j
            if "end" in line.lower() and "project gutenberg" in line.lower() and j>.75*len(lines):
                end_book_i = j
        # pass three, caught them all (check)
        if end_book_i == len(lines)-1:
            for j, line in enumerate(lines):
                if "THE END" in line and j>.9*len(lines):
                    end_book_i = j
    return lines[(start_book_i+1):(end_book_i)]


def process_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file)
    metadata['id_num'] = metadata['id'].map(lambda x: int(x[2:]))
    metadata = metadata.set_index('id_num').sort_index()
    metadata.rename(columns={'language': 'language_str'}, inplace=True)
    metadata['language'] = metadata['language_str'].map(lambda x: eval(x)[0])
    metadata['title_lower'] = metadata['title'].map(
        lambda x: x.lower() if isinstance(x, str) else x)

    return metadata


class ScoreTimeSeries(object):
    def __init__(self, pg_id, datadir):
        self.pg_id = pg_id
        self.datadir = datadir
        self.raw_ts = None
        self.diff_ts = None
        self.imf = None
        self.imf_df = None
        self.hht_df = None

    def set_raw_ts(self, _ts):
        self.raw_ts = _ts

    def get_raw_ts(self, set_prop=False, suffix='nrc_avg_scores',
                   interpolate=False, seed=False,
                   stdize=False):
        datadir = self.datadir
        pg_id = self.pg_id

        try:
            raw_ts = pd.read_csv(Path(datadir) / f'{pg_id}_{suffix}.csv',
                                 index_col=0)
        except FileNotFoundError:
            raw_ts = pd.read_csv(Path(datadir) / f'{pg_id}_{suffix}.csv.gz',
                                 index_col=0)
        if raw_ts.isna().any(axis=0).sum() > 0:
            if interpolate:
                raw_ts = raw_ts.interpolate(method='linear', limit=1)
                logging.warning('WARNING: NaN in raw TS removed using '
                                'linear interpolation.')
                assert raw_ts.isna().any(axis=0).sum() == 0
            else:
                logging.warning('WARNING: NaN detected in raw TS')

        if stdize:
            raw_ts = raw_ts.transform(standardize)
            logging.info("Time series standardized.")

        if seed is not False:
            raw_ts = raw_ts.sample(frac=1, random_state=seed)

        if set_prop:
            self.raw_ts = raw_ts
        return raw_ts

    def randomize_ts(self, seed=None, set_prop=False):
        raw_ts = self.raw_ts.sample(frac=1, random_state=seed)
        if set_prop:
            self.raw_ts = raw_ts

        return raw_ts

    # def compute_fft(self, score_col, mode='diff'):
    #     """Computes the power spectral density and the frequencies."""
    #     if mode == 'diff':
    #         if self.diff_ts is None:
    #             _diff = self.get_diff_ts()
    #         diff_ts = _diff.dropna()
    #         assert len(diff_ts) + 1 == len(_diff)
    #         ts = diff_ts[score_col]
    #     elif mode == 'orig':
    #         if self.raw_ts is None:
    #             raw_ts = self.get_raw_ts()

    #         ts = raw_ts[score_col]
    #         ts = ts - ts.mean()

    #     power = np.abs(rfft(ts.to_numpy())) ** 2
    #     freqs = rfftfreq(len(ts), 1)

    #     return freqs, power

    def get_imfs(self, score_col,
                 outputfile=None,
                 set_prop=True,
                 complete=False,
                 **kwargs):
        if self.raw_ts is None:
            ts = self.get_raw_ts()
        else:
            ts = self.raw_ts

        _, yf = ts.index, ts[score_col]

        if complete:
            imf = emd.sift.complete_ensemble_sift(yf, **kwargs)
        else:
            nensembles = kwargs.pop('nensembles', 1)
            if nensembles != 1:
                imf = emd.sift.ensemble_sift(yf, nensembles=nensembles, **kwargs)
            else:
                kwargs.pop('seed')
                imf = emd.sift.sift(yf, **kwargs)

        df = pd.DataFrame(imf, columns=[f'imf_{i}' for i in
                                        range(imf.shape[1])])

        if set_prop:
            self.imf = imf
            self.imf_df = df

        if outputfile is not None:
            logging.info('Writing IMFs...')
            df.to_csv(outputfile)
            logging.info(f'Done: see {outputfile}')

        return imf

    def get_hht(self, score_col, method='nht',
                min_freq=1e-6, max_freq=1, freq_numbins=100,
                outputfile=None,
                imf_file=None,
                sample_rate=1,
                set_prop=True,
                **kwargs):
        if imf_file is not None:
            imf = pd.read_csv(imf_file, index_col=0)
        else:
            if self.imf is None:
                imf = self.get_imfs(score_col, outputfile=None, set_prop=False,
                                    **kwargs)
            else:
                imf = self.imf

        inst_phase, inst_freq, inst_ampl = emd.spectra.frequency_transform(
            imf, sample_rate=sample_rate, method=method)

        freq_edges, freq_centers = emd.spectra.define_hist_bins(
            min_freq, max_freq, freq_numbins, 'log')

        spec_weighted = emd.spectra.hilberthuang_1d(
            inst_freq, inst_ampl, freq_edges)

        imf_hht_data = {f'inst_ampl_hist_imf_{i}': spec_weighted[:, i]
                        for i in range(spec_weighted.shape[1])}

        df = pd.DataFrame(imf_hht_data,
                          index=freq_centers).rename_axis('inst_freq_ctr')

        if set_prop:
            self.hht_df = df

        if outputfile is not None:
            logging.info('Writing HHTs...')
            df.to_csv(outputfile)
            logging.info(f'Done: See {outputfile}')

        return df

    # def get_hht_batch(self,
    #                   score_cols: ArrayLike,
    #                   outputfile=None,
    #                   **kwargs
    #                   ):
    #     """
    #     Similar to `get_hht` but creates a single file for all scores.
    #     """
    #     if self.raw_ts is None:
    #         self.get_raw_ts(set_prop=True)

    #     if kwargs is None:
    #         kwargs = {}

    #     _hht_df_list = [None] * len(score_cols)
    #     for i, score_col in enumerate(score_cols):
    #         _hht_df_list[i] = self.get_hht(
    #             score_col, outputfile=None, **kwargs)

    #     df = pd.concat(_hht_df_list, axis=0)

    #     if outputfile is not None:
    #         df.to_csv(outputfile)
    #         logging.info(f'Done: See {outputfile}')

    #     return df


class HHTData(object):
    def __init__(self, pg_id, hht_dir, score_col):
        self.pg_id = pg_id
        self.hht_dir = hht_dir
        self.score_col = score_col
        self.ts = None
        self.hht_df = None
        self.imf_df = None

    def set_ts(self, datadir, set_prop=True, suffix='nrc_avg_scores', **kwargs):
        sts = ScoreTimeSeries(self.pg_id, datadir / suffix)
        ts = sts.get_raw_ts(suffix=suffix, **kwargs)
        if set_prop:
            self.ts = ts

        return ts

    def set_hht_df(self, set_prop=True):
        hht_file = Path(self.hht_dir) / 'hht' / self.score_col / f'{self.pg_id}_hht_{self.score_col}.csv'
        hht_file = utils_g.try_gz_suffix(hht_file)

        hht_df = pd.read_csv(hht_file, float_precision='round_trip', dtype=np.float64, index_col=0)
        if set_prop:
            self.hht_df = hht_df

        return hht_df

    def set_imf_df(self, set_prop=True):
        imf_file = self.hht_dir / 'imf' / self.score_col / f'{self.pg_id}_imf_{self.score_col}.csv'
        imf_file = utils_g.try_gz_suffix(imf_file)

        imf_df = pd.read_csv(imf_file, index_col=0)
        if set_prop:
            self.imf_df = imf_df

        return imf_df


def standardize(y, ddof=1):
    return (y - y.mean()) / y.std(ddof=ddof)


class CutoffDFFxns(object):
    def __init__(self, summary_stats, datadir=None):
        self.summary_stats = summary_stats
        if datadir is None:
            self.datadir = Path('/Users/mfudolig/datadir')

    def expand_cutoff_df(self, cutoff_file, freq_info_var, imf_var):
        cutoff_df = pd.read_csv(cutoff_file, index_col=0)
        cutoff_df = cutoff_df.merge(self.summary_stats[['numwords_total']],
                                    left_index=True, right_index=True)
        cutoff_df['numwords_total'] = cutoff_df['numwords_total'].astype('int')
        for col in [col for col in cutoff_df.columns if col.startswith('rel_idx')]:
            cutoff_df[col] = cutoff_df[col].astype('Int64')

        # mode_numwords_0 refers to the original time series (not the shuffled ones)
        cutoff_df = cutoff_df.merge(
            freq_info_var[('median', 'mode_numwords_0', 'most_common')].rename(
                'period_numwords_var').to_frame(),
                left_index=True, right_index=True, how='left')
        # cutoff_df['rel_idx_pdf-var'] = cutoff_df['rel_idx'] - cutoff_df['rel_idx_var']
        
        for suffix in ['', '_var']:
            cutoff_df[f'variance{suffix}'] = cutoff_df.apply(
                lambda row: imf_var.loc[row.name, row[f'rel_idx{suffix}']] if not pd.isna(row[f'rel_idx{suffix}']) else np.nan, axis=1)

        return cutoff_df

    def get_cutoff_df(self, rescale, score_col):
        try:
            imf_var = pd.read_csv(self.datadir / f'hht_nensembles=100_resample=True/window=50_n=None_skip=50_thresh=0.7_shuffle=False/imf/{score_col}/combined/imf_variance.csv.gz', index_col=0)
        except FileNotFoundError:
            imf_var = pd.read_csv(f'~/datadir/hht_nensembles=100_resample=True/window=50_n=None_skip=50_thresh=0.7_shuffle=False/imf/{score_col}/combined/imf_variance.csv', index_col=0)
        imf_var.columns = [int(x.split('_')[-1]) for x in imf_var.columns]

        if rescale != 'None':
            freq_info_var = pd.read_csv(self.datadir / f'hht_nensembles=100_resample=True/window=50_n=None_skip=50_thresh=0.7_shuffle=False/freq_info_seed=multiple/{score_col}/combined_startidx=0_endidx=120_AGGSEEDFROMFILE_numseeds=100/cutoff_idx_true_numseeds=100_rescale={rescale}_REL_IDX_VAR_AGGSEED.csv.gz',
                                        header=[0,1,2], index_col=0)
        else:
            # ignore the first IMF order
            freq_info_var = pd.read_csv(self.datadir / f'hht_nensembles=100_resample=True/window=50_n=None_skip=50_thresh=0.7_shuffle=False/freq_info_seed=multiple/{score_col}/combined_startidx=0_endidx=120_AGGSEEDFROMFILE_numseeds=100/cutoff_idx_true_numseeds=100_rescale={rescale}_REL_IDX_VAR_EXCLFIRST_AGGSEED.csv.gz',
                                        header=[0,1,2], index_col=0)
        cutoff_df = self.expand_cutoff_df(self.datadir / f'hht_nensembles=100_resample=True/window=50_n=None_skip=50_thresh=0.7_shuffle=False/freq_info_seed=multiple/{score_col}/combined_startidx=0_endidx=120_AGGSEEDFROMFILE_numseeds=100/cutoff_idx_true_numseeds=100_rescale={rescale}.csv.gz',
                                          freq_info_var, imf_var)

        return cutoff_df

    def get_cutoff_df_dict(self, score_col):
        cutoff_df_dict = {}
        for rescale in [1, 50, 'None']:
            cutoff_df_dict[rescale] = self.get_cutoff_df(rescale, score_col)
            if rescale == 'None':
                # make sure all the 'exclfirst' overwrite the 'var' columns when rescale=None; keep the original '*var' columns as '*var_orig'
                cutoff_df_dict[rescale]['fluc_or_trend_var_orig'] = cutoff_df_dict[rescale]['fluc_or_trend_var'][:]
                cutoff_df_dict[rescale]['rel_idx_var_orig'] = cutoff_df_dict[rescale]['rel_idx_var'][:]
                cutoff_df_dict[rescale]['fluc_or_trend_var'] = cutoff_df_dict[rescale]['fluc_or_trend_var_exclfirst'][:]
                cutoff_df_dict[rescale]['rel_idx_var'] = cutoff_df_dict[rescale]['rel_idx_var_exclfirst'][:]

        return cutoff_df_dict


class LCCMethods(object):
    def __init__(self, metadata, cutoff_df_dict, init=True):
        self.metadata = metadata
        self.cutoff_df_dict = cutoff_df_dict
        self.metadata_lcc = None
        self.lcc_pgids = None
        self.cutoff_df_dict_lcc = None

        if init:
            self.set_metadata_lcc()
            self.set_lcc_pgids()
            self.generate_cutoff_df_dict_lcc()

    def set_metadata_lcc(self, as_prop=True):
        metadata_lcc = self.metadata.explode('lcc_main')
        metadata_lcc['lcc_broad'] = metadata_lcc['lcc_main'].map(lambda x: x[0])

        if as_prop:
            self.metadata_lcc = metadata_lcc
        return metadata_lcc

    def set_lcc_pgids(self, as_prop=True):
        if self.metadata_lcc is None:
            metadata_lcc = self.set_metadata_lcc(as_prop=False)
        else:
            metadata_lcc = self.metadata_lcc
        lcc_pgids = {}
        for lcc_type in ['lcc_broad', 'lcc_main']:
            lcc_pgids[lcc_type] = {}
            for lcc in metadata_lcc[lcc_type].unique():
                lcc_pgids[lcc_type][lcc] = metadata_lcc.query(f'{lcc_type} == "{lcc}"').index.unique()
        
        if as_prop:
            self.lcc_pgids = lcc_pgids

        return lcc_pgids

    def generate_cutoff_df_dict_lcc(self, as_prop=True):
        if self.lcc_pgids is None:
            lcc_pgids = self.set_lcc_pgids(as_prop=False)
        else:
            lcc_pgids = self.lcc_pgids

        if self.metadata_lcc is None:
            metadata_lcc = self.set_metadata_lcc(as_prop=False)
        else:
            metadata_lcc = self.metadata_lcc

        cutoff_df_dict_lcc = {}
        for lcc_type in ['lcc_broad', 'lcc_main']:
            cutoff_df_dict_lcc[lcc_type] = {}
            for lcc in metadata_lcc[lcc_type].value_counts().index:
                cutoff_df_dict_lcc[lcc_type][lcc] = {}
                for k, v in self.cutoff_df_dict.items():
                    cutoff_df_dict_lcc[lcc_type][lcc][k] = v.loc[
                        v.index.intersection(lcc_pgids[lcc_type][lcc])]
        
        if as_prop:
            self.cutoff_df_dict_lcc = cutoff_df_dict_lcc

        return cutoff_df_dict_lcc