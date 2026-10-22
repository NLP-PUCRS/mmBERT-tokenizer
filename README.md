# Tokenizer Impact on Multilingual to Portuguese Transfer in ModernBERT-Based Models

This repository contains the code and experiments for evaluating the impact of vocabulary trimming on multilingual models transferred to Portuguese, as described in the paper.

## ToC

1. [Cite](#cite)
1. [Project Structure](#project-structure)
1. [Datasets](#datasets)
1. [Installation](#installation)
1. [Creating Trimmed Tokenizers](#creating-trimmed-tokenizers)
1. [Extrinsic Evaluation (Downstream Tasks)](#extrinsic-evaluation-downstream-tasks)
1. [Results Format](#results-format)

## Cite

- [BRACIS 2026](https://bracis.sbc.org.br/2026/)

	```bibtex
	@inproceedings{ACCORSI:BRACIS:27,
		title={{Tokenizer Impact on Multilingual to Portuguese Transfer in ModernBERT-Based Models}},
		author={Arthur Turcati Accorsi and Gustavo Losch do Amaral and Carlos Falcao de Azevedo Gomes and Dalvan Griebler},
		editor={Mariana Recamonde-Mendoza and Rodrigo C. Barros},
		booktitle={Intelligent Systems},
		year={2027},
		publisher={Springer Nature Switzerland},
		address={Cham},
		pages={517-531},
		isbn={978-3-032-39892-5},
		doi={10.1007/978-3-032-39892-5_34},
		url={https://doi.org/10.1007/978-3-032-39892-5_34}
	}
	```

## Project Structure

- `src/tokenizer_adaptation/`: Scripts for vocabulary analysis and tokenizer trimming.
- `src/downstream_tasks/`: Fine-tuning pipeline, including data utilities, metrics, and trainer orchestration.
- `notebooks/`: Exploratory and demonstration notebooks.

## Datasets

We use the following Portuguese datasets for fine-tuning and evaluation:

- **[ASSIN2](https://sites.google.com/view/assin2/)**: Used for STS and RTE tasks.
- **[LeNER-Br](https://teodecampos.github.io/LeNER-Br/)**: Used for NER in the legal domain.

## Installation

Install the required dependencies using pip:

```shell
pip install -r requirements.txt
```

Copy the `.env.example` file to create your own `.env` file. You will need to add your Hugging Face read-only token and configure the variables for both the tokenizer trimming and downstream evaluation processes.

```shell
cp .env.example .env
```

## Creating Trimmed Tokenizers

You can create a language-specific trimmed tokenizer by analyzing a corpus (e.g., Wikipedia) and selecting the most frequent tokens while preserving essential multilingual base characters and special tokens.

### Configuration

Modify the tokenizer variables at `.env`:

```txt
# Original multilingual model path
SOURCE_TOKENIZER=models/mmbert
# Output directory for the trimmed model
OUTPUT_DIR=models/mmbert_pt_32k
# Desired vocabulary size
MAX_VOCAB_SIZE=32768
# Number of samples for frequency analysis
SAMPLE_SIZE=110000
# Corpus dataset
DATASET_NAME=wikimedia/wikipedia
# Dataset language configuration
DATASET_CONFIG=20231101.pt
```

### Execution

Run the script from the root directory:

```shell
python src/tokenizer_adaptation/create_trimmed_tokenizer.py
```

## Extrinsic Evaluation (Downstream Tasks)

To fine-tune and evaluate models on Portuguese downstream tasks:

1. **Configure Models**: Add your models to the `MODELS_DICT` in `src/experiments/downstream_tasks/config.py`.
2. **Set Hyperparameters**: Adjust the Optuna search space in the `optuna_hp_space` function within the same file.
3. **Run Experiment**:

    ```shell
    # Direct execution
    python src/downstream_tasks/main.py

    # Using Slurm
    sbatch run_downstream.sh
    ```

    The pipeline automatically performs hyperparameter optimization using Optuna and runs multiple training replications across different seeds.

## Results Format

Upon completion of the experiments, the results are stored in `results/`.

### Individual Task Summaries

Each task (NER, STS, RTE) and model combination generates a JSON summary file (e.g., `summary_<model>_<timestamp>.json`) that contain model metadata (parameters, name), best hyperparameters, detailed metrics for each seed replication and aggregated mean and standard deviation for all evaluation metrics.

### Aggregated Results

You can use `src/result_parsing.py` to aggregate all individual task summaries into a single CSV file that contains the model name, timestamp of experiment execution, mean and std of downstream result metrics over the replication seeds:

```shell
python src/result_parsing.py
```
