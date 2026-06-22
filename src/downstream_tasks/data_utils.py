from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def load_all_datasets() -> tuple[DatasetDict, DatasetDict, DatasetDict, list[str]]:
    """Loads and splits LeNER-Br and ASSIN2 datasets for NER, STS, and RTE tasks."""

    # Load LeNER-Br from Hugging Face
    raw_ner = load_dataset("peluz/lener_br", revision="refs/convert/parquet")
    ner_ds = DatasetDict({
        "train": raw_ner["train"],
        "validation": raw_ner["validation"],
        "test": raw_ner["test"]
    })
    
    # Load ASSIN2 for similarity and entailment
    assin2 = load_dataset("assin2")
    sts_ds = DatasetDict({
        "train": assin2["train"],
        "validation": assin2["validation"],
        "test": assin2["test"]
    })
    rte_ds = DatasetDict({
        "train": assin2["train"],
        "validation": assin2["validation"],
        "test": assin2["test"]
    })
    
    # Get the unique list of NER tags
    label_list = ner_ds["train"].features["ner_tags"].feature.names
    
    return ner_ds, sts_ds, rte_ds, label_list

def preprocess_sts(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenizes premise-hypothesis pairs and extracts relatedness scores for STS."""
    enc = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        return_overflowing_tokens=True
    )
    # Convert scores to float for regression task
    enc["labels"] = [float(x) for x in examples["relatedness_score"]]
    return enc

def preprocess_rte(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenizes premise-hypothesis pairs and extracts entailment judgments for RTE."""
    enc = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        return_overflowing_tokens=True
    )
    enc["labels"] = examples["entailment_judgment"]
    return enc

def preprocess_ner(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenizes word sequences and aligns NER tags with the resulting tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_overflowing_tokens=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Map special tokens to -100 to ignore them in loss
            if word_idx is None:
                label_ids.append(-100)
            # Assign label only to the first token of a split word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs