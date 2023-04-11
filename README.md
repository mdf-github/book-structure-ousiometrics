# Generating data for "A decomposition of book structure through ousiometric fluctuations in cumulative word-time"<sup>[[1]](#1)</sup>

### Installing a conda environment

Clone this repo into your machine. Open a terminal in the folder containing all the downloaded repo files, and create a conda environment (which we name `ousiometrics`) using

```
conda env create --file ousiometrics.yml
conda activate ousiometrics
```

### Downloading text from Gutenberg
Download a text file from Project Gutenberg using the instructions from their website: https://www.gutenberg.org/help/mirroring.html.

As an example, we placed a downloaded file from Project Gutenberg in the `./demo/gutenberg_txt/` folder.

The Project Gutenberg IDs used in the study are given in `PG_IDs.csv`.

### Removing headers

We want to remove the headers of the files from Gutenberg. Assuming the downloaded text files from Gutenberg (with extensions beginning with `*.txt`) are in `$SRCDIR_ORIG` and you want to place the processed files in `$SRCDIR_CLEAN`, run

`python cleanup_gutenberg_headers.py $SRCDIR_ORIG $SRCDIR_CLEAN`

For the demonstration, we use `./demo/gutenberg_txt` as `$SRCDIR_ORIG` and `./demo/gutenberg_txt_clean` as `$SRCDIR_CLEAN`. The file `cleanup_gutenberg_headers.py` was adapted from the code in [[2]](#1).

### Generating the time series

The time series can be generated for a given window size $N_w$ and skip size $N_s$. In our analysis, we used $N_w=50$ and $N_s=50$. If we want to put the resulting files 
in `./demo/outputdir`, the corresponding scores for every window can be generated by

```
python generate_texttrends_files.py ./demo/gutenberg_txt_clean/PG1342_text.txt ./demo/outputdir --N_w 50 --N_s 50 --overwrite
```

The `--overwrite` argument is included to overwrite the relevant output files in `./demo/*/`.

To generate the scores for shuffled text, one can use the additional arguments `--shuffle` and `--seed`. In this case, we set the seed to be `42`:

```
python generate_texttrends_files.py ./demo/gutenberg_txt_clean/PG1342_text.txt ./demo/outputdir --N_w 50 --N_s 50 --shuffle --seed 42 --overwrite
```

Note that new subfolders inside `./outputdir` are generated automatically.

### Generating the IMFs

To compute for the IMFs, the `emd` module [[3]](#1) has to be installed from the folder in this repo. This is because the `ensemble_sift` function was modified to allow for a seed as a keyword argument.

To install the modified `emd` module:

```
cd ./emd
pip install -r requirements.txt
pip install .
```

Return to the folder containing the file `get_hht_freqs.py`. To obtain the IMF files for the time series corresponding to a PG ID in some `$TS_FOLDER` (e.g., `1342` in folder `./demo/outputdir/window=50_n=None_skip=50_thresh=0.7_shuffle=False/`), run
```
python get_hht_freqs.py ./demo/outputdir/window=50_n=None_skip=50_thresh=0.7_shuffle=False/ 1342 --overwrite
```
The output subdirectories `hht/` and `imf/` containing the HHT and IMF-related files will be in `$TS_FOLDER`. Columns ending with a `_<number>` correspond to a given IMF order, with the last IMF computed by `emd` corresponding to the trend. Note that in Python, the first IMF is labeled `0` (i.e., `imf_0`, etc).

**References**

<a id="1">[1]</a> 
M.I. Fudolig, T. Alshaabi, K. Cramer, C.M. Danforth, and P.S. Dodds. (2023). “A decomposition of book structure through ousiometric fluctuations in cumulative word-time,” *Humanities and Social Sciences Communications* (in press), https://arxiv.org/abs/2208.09496.

<a id="2">[2]</a> 
Gerlach, M., & Font-Clos, F. (2020). A standardized Project Gutenberg corpus for statistical analysis of natural language and quantitative linguistics. Entropy, 22(1), 126. (Code in https://github.com/pgcorpus/gutenberg)

<a id="3">[3]</a> 
https://emd.readthedocs.io/en/v0.4.0/
