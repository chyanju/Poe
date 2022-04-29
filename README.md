# Poe: Visualization Question Answering Using Introspective Program Synthesis

<div align="left">
  This project is powered by <img src="./resources/trinity_edge_matrix.png" width=30> Trinity-Edge (<a href="https://github.com/chyanju/Trinity-Edge">https://github.com/chyanju/Trinity-Edge</a>).
</div>
## [Artifact Evaluation] Poe in Colab

1. To set up the machine learning model used in the artifact, we provide a colab notebook that covers every step and reproduces the procedure. Please check it out here: [link](./public_PLDI22AE_Poe_TaPas_on_VisQA.ipynb).
2. To reuse the tool and make modifications to existing benchmarks, or even create your own benchmarks, we provide a colab notebook for you to easily do that. Please check it out here: [link](public_PLDI22AE_Poe_make_benchmark.ipynb).

## [Artifact Evaluation] Getting Started Guide

A Linux (or Unix-like) machine is recommended since this is where the tool is tested upon.

1. Clone this repo.

2. Resolve the dependencies as stated in the "Dependencies" section. You can skip dependencies that are needed for processing the dataset, because the pre-processed dataset is already included in the repo. If you are interested in the dataset pre-processing, you are welcomed to read "Dataset & Paraparation" section. But dataset processing is not included in our proposed artifact evaluation plan.

3. Try to issue the following simple command from the root of this repo:

   ```bash
   python ./test_TaPas_on_VisQA_benchmark.py --benchmark Q0f532 --dsl meta_visqa --skeletons visqa_simple --strategy TaPas_C --expected-only
   ```

   This will perform a simple run of the tool on a benchmark `Q0f532` and generate a result and report. If your environment is correctly configured, you should be able to see something like the following:

   ```bash
   # parsed arguments: Namespace(benchmark='Q0f532', dsl='meta_visqa', expected_only=True, fallback='none', mode='full', skeletons='visqa_simple', strategy='TaPas_C', timeout=0)
   # loading benchmark...
   # table keywords: {'13', '20', '43', '44', '2', '83', '81', '9', '50', 'britain', '84', 'greece', '28', '24', 'remain', 'india', '29', '37', '45', 'u.s.', 'red', 'lebanon', 'worsen', 'country', '32', '23', 'tunisia', 'percentage', 'improve', 'germany', 'pakistan', 'color', 'spain', 'brazil', 'same', 'italy', '49', '34', 'blue', 'czech', 'china', '12', 'jordan', '15', 'egypt', 'russia', '26', 'orange', '52', '10', '31', '33', 'poland', '27', '35', '5', '47', 'france', '22', 'mexico', '16', 'japan', '18', '75', 'opinion', '51', '40', '60', 'rep.', '25', 'turkey'}
   # input type: [dtype('O'), dtype('O'), dtype('int64'), dtype('O')]
   # input is:
          Country          opinion  percentage   color
   0       Brazil          Improve          84    blue
   1       Brazil  Remain the same          12  orange
   2       Brazil           Worsen           5     red
   3        China          Improve          83    blue
   4        China  Remain the same           9  orange
   ..         ...              ...         ...     ...
   58  Czech Rep.  Remain the same          27  orange
   59  Czech Rep.           Worsen          60     red
   60      Greece          Improve           9    blue
   61      Greece  Remain the same          10  orange
   62      Greece           Worsen          81     red
   
   [63 rows x 4 columns]
   # query is: which country's economy will get most worse over next 12 months?
   # expected output type:[dtype('O')]
   # expected output is:
      ANSWER
   0  Greece
   # inferred DSL terminals:
     # ConstVal: ['Worsen@Str', 'Country@Str', '12@Int', '<NULL>']
        # cmap: [('worse', 'Worsen'), (12, 12), ('country', 'Country')]
     # AggrFunc: ['max', '<NULL>']
        # amap: [('most', 'max')]
     # NumFunc: ['<NULL>']
        # nmap: []
     # BoolFunc: ['==', '<NULL>']
        # bmap: [(None, '==')]
     # IndFunc: ['eqmax', '<NULL>']
        # imap: [('most', 'eqmax')]
   # loading skeleton list...
   
   # ========== candidate program report ========== #
   # (t=0.00) i=0, candidate=[{'ANSWER': 'Greece'}]
     # found 1 program(s)
       # SelectCol(SelectRow0(SelectRow1(@param0, ==, 1, Worsen@Str), eqmax, 2), ['0'])
         --> ['SelectCol', ['SelectRow0', ['SelectRow1', 'input@0', '==', 'opinion', 'Worsen'], 'eqmax', 'percentage'], ['Country']]
   
   # ========== review report ========== #
   # top-1, score: 1.35, answer: ['Greece']
     # tprog: SelectCol(SelectRow0(SelectRow1(@param0, ==, 1, Worsen@Str), eqmax, 2), ['0'])
   ```

4. If nothing goes wrong in the previous steps, the tool should be ready for use. Otherwise, please double-check all the required dependencies again.

## Dependencies

Please refer to [this](public_PLDI22AE_Poe_make_benchmark.ipynb) colab notebook for step-by-step environment configuration. You can also consider directly using the colab version.

The major functionalities of the tool depend on the following packages:

- Trinity-Edge ([https://github.com/chyanju/Trinity-Edge](https://github.com/chyanju/Trinity-Edge))
  - Including all its required dependencies.
- spaCy ([https://spacy.io/](https://spacy.io/))
  - Version Tested: 3.1.2
  - Execute the following to download extra corpus after installing spaCy:
    - `python -m spacy download en_core_web_sm`
    - `python -m spacy download en_core_web_lg`
- NLTK ([https://www.nltk.org/](https://www.nltk.org/))
- simplejson ([https://github.com/simplejson/simplejson](https://github.com/simplejson/simplejson))
- python-tabulate ([https://github.com/astanin/python-tabulate](https://github.com/astanin/python-tabulate))

- Node.js ([https://nodejs.org/](https://nodejs.org/))
- Vega-Lite >= 5.1.1 ([https://vega.github.io/vega-lite/](https://vega.github.io/vega-lite/))
  - This should be installed via `npm`, run this in the repo root: `npm install vega-lite`. Note that this will install vega as a non-global library in the repo root.
  - Need the command `vl2svg`: [https://vega.github.io/vega-lite/usage/compile.html](https://vega.github.io/vega-lite/usage/compile.html)

## Dataset & Preparation

The dataset used in this tool is pre-processed and included in the repo. You can find it here: `benchmarks/VisQA/shared/tapas_on_visqa_dataset.pkl`. To use the tool for evaluation on the dataset, you can skip this section and directly use the processed dataset included.

The following notes show procedures for dataset construction/adaptation from the original VisQA ([https://github.com/dhkim16/VisQA-release](https://github.com/dhkim16/VisQA-release)) dataset. 

- Question types are classified as follows:
  - `retrieval`: e.g., "what is X value of Y?" or "what X has less than A value of Y"; <u>answers are fetched directly from the table cells with absolute condition</u>.
  - `assertion`: e.g., "does X have higher value than Y?"; <u>answers are either "yes" or "no"</u>.
  - `selection`: e.g., "does X or Y have Z?" or "does X have more than A or B value"; <u>answers are chosen from the question</u>.
  - `comparison`: e.g., "what is the difference between X and Y?"; <u>answers are about concrete outcome not shown in either question or table cells</u>.
  - `aggregation`: e.g., "what is the sum/averaged of X?" or "what is the 4th highest value of X?"; <u>answers are fetched with relative global condition</u>.
  - `combination`: e.g., "does X have A more value than average?"; question contains multiple types; this is usually pretty hard.
  - `other`: questions that do not belong to any of the above types or questions that require special attentions; e.g., "what does x axis represent?" or "which racehas neither the highest nor lowest percentage?".
- The dataset used for benchmarking is from VisQA ([https://github.com/dhkim16/VisQA-release](https://github.com/dhkim16/VisQA-release)). Some formatting issues are fixed on top of the original dataset to ensure the correctness of rendered table extraction, including:
  - `kong:76`: Quote thousand-separated (`,`) numbers as string so that they can be correctly read by `pandas` and recovered by the parsing module.
  - `d3`: Added missing matching from `chartName` in `qadata.json` to the actual file name, specifically:
    - `d3_12` -> `multi_line`
    - `d3_4` -> `4702_0`
    - `d3_1` -> `2756_0`
    - `d3_10` -> `3001_0`
    - `d3_11` -> `3001_1`
    - `d3_3` -> `7920_0`
    - `d3_5` -> `2998_0`
    - `d3_6` -> `8126_0`
    - `d3_7` -> `8125_1`
    - `d3_8` -> `8125_2`
    - `d3_9` -> `8125_4`
  - `d3/10.json` spec: change type of `x` from `temporal` to `nominal` for correct rendering given normalized table.
  - `d3/12.json` spec: change type of `x` from `temporal` to `nominal` for correct rendering given normalized table.
  - `vega-lite-example-gallery/line.json` spec: change type of `x` from `temporal` to `nominal` for correct rendering given normalized table.
  - For line charts, since one line is one stroke and current version of vega only attaches the first label to the whole stroke that contains multiple points, as a fix we change the chart type from "line" to "point" if there's any, so as to make sure we extract the full data for the rendered table. The infected specs are:
    - `vega-lite-example-gallery/line.json`
    - `vega-lite-example-gallery/line_color.json`
    - `d3/10.json`
    - `d3/11.json`
    - `d3/12.json`
- Some answers are not aligned with the values from the table, which are corrected. See the progress sheet for more details.
- You can run `build-visqa-dataset.ipynb` to re-build VisQA dataset (even though it's already included). Building a dataset is just running some computations beforehand (e.g., rendering tables) so as to shorten the time the user should wait during testing.
- The `tapas-on-visqa-helper.ipynb` provides scripts that extend the current TaPas predictions into top-k form, by applying different extension strategies. For details of different strategies, please read the corresponding comments in the notebook.

## Usage

The main entrance of the tool is `test_TaPas_on_VisQA_benchmark.py`.

```bash
usage: test_TaPas_on_VisQA_benchmark.py [-h] [-i DATASET] [-b BENCHMARK] 
                                        [-d {test_min,meta_visqa}] 
                                        [-s {test_min,visqa_simple,visqa_normal}] 
                                        [-g {TaPas_C}] [-f {none,auto}] [-t TIMEOUT]
                                        [-m {full,optimal-only,abstract-only}] 
                                        [--expected-only]

optional arguments:
  -h, --help            show this help message and exit
  -i DATASET, --dataset DATASET
                        the input dataset, default: benchmarks/VisQA/shared/tapas_on_visqa_dataset.pkl
  -b BENCHMARK, --benchmark BENCHMARK
                        6-charactered benchmark id, default: Q0f532
  -d {test_min,meta_visqa}, --dsl {test_min,meta_visqa}
                        DSL definition to use, default: test_min
  -s {test_min,visqa_simple,visqa_normal}, --skeletons {test_min,visqa_simple,visqa_normal}
                        skeleton list to use, default: test_min
  -g {TaPas_C}, --strategy {TaPas_C}
                        candidate generation strategy to apply, default: TaPas_C
  -f {none,auto}, --fallback {none,auto}
                        specify fallback strategy, default: none
  -t TIMEOUT, --timeout TIMEOUT
                        timeout in seconds, default: 0 (no timeout)
  -m {full,optimal-only,abstract-only}, --mode {full,optimal-only,abstract-only}
                        ablation mode to use, default: full
  --expected-only       whether or not to only process the expected answer (for debugging), default: False
```

Here are some example commands:

```bash
# testing a benchmark with its expected output
python ./test_TaPas_on_VisQA_benchmark.py --benchmark Q0f532 --dsl meta_visqa --skeletons visqa_simple --strategy TaPas_C --expected-only

# most common setting with simple skeleton list, no fallback policy
python ./test_TaPas_on_VisQA_benchmark.py --benchmark Q15586 --dsl meta_visqa --skeletons visqa_simple --strategy TaPas_C

# most common setting with normal skeleton list, with fallback policy
python ./test_TaPas_on_VisQA_benchmark.py --benchmark Q0f532 --dsl meta_visqa --skeletons visqa_normal --strategy TaPas_C --fallback auto --timeout 0 --mode full

# most common setting with normal skeleton list, with fallback policy, 5min timeout, no optimal synthesis
python ./test_TaPas_on_VisQA_benchmark.py --benchmark Q0f532 --dsl meta_visqa --skeletons visqa_normal --strategy TaPas_C --fallback auto --timeout 300 --mode abstract-only
```

## [Artifact Evaluation] Step-by-Step Instructions

The main results of the paper are shown by Figure 7 and Table 2. The following sub-sections elaborate how one can reproduce them. In addition to the major results, we also show how to reproduce other results e.g. Table 1.

### Collecting Results

Before generating any reported results, run the following commands first to solve all the questions with different settings respectively:

```bash
# Poe full-fledged version
./run.sh full 0 628 300

# Poe with optimal synthesis only
./run.sh optimal-only 0 628 300

# Poe with abstract synthesis only
./run.sh abstract-only 0 628 300
```

Each command will generate log files (one log file for each benchmark) in naming conventions `logs/<benchmark>_<mode>.log` where `<benchmark>` corresponds to benchmark id and `<mode>` corresponds to one of the variants of Poe (`full`, `optimal-only` or `abstract-only`).

All the follow-up results will be based on the analysis of the above generated log files.

#### Scaling up the experiments

***Note: Each command may take hours to finish (depending on hardware environment).*** To speed up the evaluation, you can split one command in multiple processes (or on multiple vms). For example the first command can be split into the following:

```bash
# Poe full-fledged version, parallel run
./run.sh full 0 200 300
./run.sh full 201 400 300
./run.sh full 401 628 300
```

And upon finishing running, collect all the log files into one location.

#### Using provided logs

In case there are still issues running the tool, we've also provided the experiment logs for the above commands. You can find them in `logs/`, and use them to proceed with the analysis instead.

### Results in Table 2

Make sure you have finished the "Collecting Results" before performing the following steps.

1. In Jupyter Notebook/Lab, open the notebook `analyze-results-Poe.ipynb`. 
2. In the second cell, uncomment the statement for `full`, `abstract-only` or `optimal-only` for analyzing Poe, Poe-A and Poe-O respectively. If you are analyzing the log files collected by yourself, change the statement accordingly to point to the corresponding folder and file pattern.
3. Execute the whole notebook till the end, and you will see `acc` value, which corresponds to the `solved` number from the paper. The `n_timeout` value corresponds to the `#timeout` reported in the paper.

Note that this automated comparison of answers in natural language may lead to inaccurate comparison results, and we eventually went through a manual inspection of all the answers one by one to get the precise numbers. Therefore, the automated computed numbers here may be slightly different to the reported numbers, but such difference is usually within 10.

### Results in Figure 7

Make sure you have finished the "Collecting Results" before performing the following steps.

To reproduce Poe-1, Poe-3 and Poe-5 respectively in Figure 7, you can follow similar steps reproducing Table 2:

1. In Jupyter Notebook/Lab, open the notebook `analyze-results-Poe.ipynb`. 
2. In the second cell, uncomment the statement for `full` for analyzing Poe (full-fledged version). If you are analyzing the log files collected by yourself, change the statement accordingly to point to the corresponding folder and file pattern.
3. Search for the key assignment of the constant `N_TOP` in the notebook, and change it to `N_TOP=1` for Poe-1, `N_TOP=3` for Poe-3 and `N_TOP=5` for Poe-5.
4. Execute the whole notebook till the end, and you will see `acc` value, which corresponds to the `solved` number from the paper.

Note that this automated comparison of answers in natural language may lead to inaccurate comparison results, and we eventually went through a manual inspection of all the answers one by one to get the precise numbers. Therefore, the automated computed numbers here may be slightly different to the reported numbers, but such difference is usually within 10.

To reproduce TaPas:

1. In Jupyter Notebook/Lab, open the notebook `analyze-results-TaPas.ipynb`. 
2. Execute all cells to the end, the `acc` corresponds to the TaPas `solved` number reported by the paper.

There's no automated script to reproduce the `solved` value of VisQA, since we manually inspect the results. Please refer to the benchmark worksheet for details (`logs/benchmark-worksheet.csv`). The original logs of VisQA can be found at `logs/visqa/`.

### Results in Table 1

We manually categorize every benchmark with different question types. There's no automated scripts for such manual inspection. We provide a benchmark worksheet for details (`logs/benchmark-worksheet.csv`). You can easily reproduce the statistics in the table using basic table/sheet processing softwares.

### Other Results

- For flip rate computation claimed in section 7.3, you can easily compute them using the benchmark worksheet `logs/benchmark-worksheet.csv`, where column `Poe check` , `TaPas check` and `VisQA check` indicate correctness of every tool's output compared to the desired answer respectively. `0` means wrong and `1` means correct.
- You can check out the VisQA repo ([https://github.com/dhkim16/VisQA-release](https://github.com/dhkim16/VisQA-release)) and the TaPas repo ([https://github.com/google-research/tapas](https://github.com/google-research/tapas)) for setting up an environment for their tools. In this repo, we provide logs for running their tools on the processed benchmarks. See `logs/` for original log files or the benchmark worksheet for more detailed analysis.

### Reusability

Poe provides an easy-to-use entrance. The internal framework structure follows that of Trinity ([https://github.com/fredfeng/Trinity](https://github.com/fredfeng/Trinity)) and Trinity-Edge ([https://github.com/chyanju/Trinity-Edge](https://github.com/chyanju/Trinity-Edge)), which are designed for better extensibility and well documented. One can easily build upon Poe following similar procedure of Trinity and Trinity-Edge, and will also find it easy to change evaluation settings according to the "Usage" section.

## Citation

Please check back later for bibtex.
