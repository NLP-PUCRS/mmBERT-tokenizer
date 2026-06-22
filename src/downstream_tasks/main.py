from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForSequenceClassification, 
    DataCollatorForTokenClassification,
    DataCollatorWithPadding
)
from config import MODELS_DICT
from data_utils import load_all_datasets, preprocess_ner, preprocess_rte, preprocess_sts
from metrics import get_compute_metrics_ner, compute_metrics_rte, compute_metrics_sts
from trainer_utils import train_and_evaluate
from datetime import datetime
from dotenv import load_dotenv

def main() -> None:
    """Entry point to execute fine-tuning and evaluation on NER, STS, and RTE for all configured models."""
    load_dotenv()
    # Load all datasets and the label list for NER
    ner_ds, sts_ds, rte_ds, label_list = load_all_datasets()
    num_ner_labels = len(label_list)
    ner_compute_metrics_fn = get_compute_metrics_ner(label_list)
    
    results = {}
    
    # Iterate through models defined in config.py
    for run_name, model_name in MODELS_DICT.items():
        print(f"Executing: {run_name} ({model_name})")
        timestamp = datetime.now()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Phase 1: NER Task
        print("Preparing NER data")
        tokenized_ner = ner_ds.map(lambda x: preprocess_ner(x, tokenizer), batched=True)
        tokenized_ner = tokenized_ner.filter(lambda x: len(x['input_ids']) > 0)
        data_collator_ner = DataCollatorForTokenClassification(tokenizer)
        
        def model_init_ner():
            return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_ner_labels)
            
        ner_metrics = train_and_evaluate(
            run_name=run_name,
            task_name="ner",
            timestamp=timestamp,
            model_init_fn=model_init_ner,
            train_ds=tokenized_ner["train"],
            eval_ds=tokenized_ner["validation"],
            test_ds=tokenized_ner["test"],
            compute_metrics_fn=ner_compute_metrics_fn,
            metric_name="f1",
            data_collator=data_collator_ner,
            direction="maximize",
            gradient_checkpointing=False
        )
        
        # Phase 2: STS Task
        print("Preparing STS data")
        tokenized_sts = sts_ds.map(lambda x: preprocess_sts(x, tokenizer), batched=True, remove_columns=sts_ds["train"].column_names)
        data_collator_sts_rte = DataCollatorWithPadding(tokenizer)

        def model_init_sts():
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression", device_map="auto")
            for param in model.parameters(): param.requires_grad = True
            model.enable_input_require_grads()
            return model
            
        sts_metrics = train_and_evaluate(
            run_name=run_name,
            task_name="sts",
            timestamp=timestamp,
            model_init_fn=model_init_sts,
            train_ds=tokenized_sts["train"],
            eval_ds=tokenized_sts["validation"],
            test_ds=tokenized_sts["test"],
            compute_metrics_fn=compute_metrics_sts,
            metric_name="pearson",
            data_collator=data_collator_sts_rte,
            direction="maximize",
            gradient_checkpointing=True
        )
        
        # Phase 3: RTE Task
        print("Preparing RTE data")
        tokenized_rte = rte_ds.map(lambda x: preprocess_rte(x, tokenizer), batched=True, remove_columns=rte_ds["train"].column_names)
        
        def model_init_rte():
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map="auto")
            for param in model.parameters(): param.requires_grad = True
            model.enable_input_require_grads()
            return model
            
        rte_metrics = train_and_evaluate(
            run_name=run_name,
            task_name="rte",
            timestamp=timestamp,
            model_init_fn=model_init_rte,
            train_ds=tokenized_rte["train"],
            eval_ds=tokenized_rte["validation"],
            test_ds=tokenized_rte["test"],
            compute_metrics_fn=compute_metrics_rte,
            metric_name="f1_macro",
            data_collator=data_collator_sts_rte,
            direction="maximize",
            gradient_checkpointing=True
        )
        
        results[run_name] = {
            "model": model_name,
            "NER": ner_metrics,
            "STS": sts_metrics,
            "RTE": rte_metrics
        }

    # Print summary of results
    for model, metrics in results.items():
        print(f"\n{model}")
        print(" NER F1:", f"{metrics['NER'].get('mean_f1', 0):.4f}")
        print(" STS Pearson:", f"{metrics['STS'].get('mean_pearson', 0):.4f}")
        print(" RTE Macro F1:", f"{metrics['RTE'].get('mean_f1_macro', 0):.4f}")

if __name__ == "__main__":
    main()
