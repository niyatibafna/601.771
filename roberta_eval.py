"""
Train a RoBERTa model on the SST-2 dataset for classification.
"""
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from datasets import load_dataset
import evaluate

# Get dataset
ds = load_dataset("stanfordnlp/sst2")

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

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
    num_train_epochs=3,
    logging_dir="logs/roberta/",
    report_to="tensorboard",
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    output_dir="checkpoints/roberta/",
    resume_from_checkpoint=True
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
# trainer.train()

# Put the model in evaluation mode
model.eval()
# Evaluate the model on the test set
results = trainer.evaluate(tokenized_ds["test"])
print(results)

# Save the model
# model.save_pretrained("best/roberta/")
