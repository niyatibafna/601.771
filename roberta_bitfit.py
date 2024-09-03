"""
Train a RoBERTa model with LoRA on the SST-2 dataset for classification.
"""
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

# Get dataset
ds = load_dataset("stanfordnlp/sst2")

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Freeze all non-bias, non-classifier parameters
for name, param in model.named_parameters():
    if "classifier" not in name and "bias" not in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    status = "Trainable" if param.requires_grad else "Frozen"
    print(f"{name}: {status}")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=12,
    num_train_epochs=2,
    logging_dir="logs/roberta_bitfit/",
    report_to="tensorboard",
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    output_dir="checkpoints/roberta_bitfit/",
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(tokenized_ds["test"])
print(results)

# Save the model
model.save_pretrained("best/roberta_bitfit/")
