import os
import multiprocessing
import numpy as np
import torch
import transformers

from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# HYPERPARAMS
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
LR_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# load data
print("loading dataset")
dataset = load_dataset(
    "text", data_files={"train": "train.txt", "test": "test.txt"}, cache_dir="cache"
)

tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base", use_fast=True, do_lower_case=False, max_len=MAX_SEQ_LEN
)
model = RobertaForMaskedLM.from_pretrained("roberta-base")


def tokenize_function(row):
    return tokenizer(
        row["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_special_tokens_mask=True,
    )


column_names = dataset["train"].column_names

dataset["train"] = dataset["train"].map(
    tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names,
)

dataset["test"] = dataset["test"].map(
    tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


steps_per_epoch = int(len(dataset["train"]) / TRAIN_BATCH_SIZE)

training_args = TrainingArguments(
    output_dir="./output",
    logging_dir="./LMlogs",
    num_train_epochs=2,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=LR_WARMUP_STEPS,
    save_steps=steps_per_epoch,
    save_total_limit=3,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("fine-tuned")  # save your custom model
