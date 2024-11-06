from datasets import load_dataset
from evaluate import load
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, TrainingArguments, Trainer
import torch

# Determine if GPU is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)
print(torch.cuda.is_available())
print(f"Using device: {device}")

# Load dataset
ds = load_dataset("Kyrmasch/sKQuAD")

# Check for the existence of validation split
if "validation" not in ds:
    ds = ds["train"].train_test_split(test_size=0.1)
    ds["train"], ds["validation"] = ds["train"], ds["test"]

# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large')
model.to(device)

# Print dataset info
print(ds["train"].column_names)

def preprocess_function(examples):
    questions = examples['question']
    contexts = examples['context']
    answers = examples['answer']

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = tokenized_examples.pop("offset_mapping")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_index = sample_mapping[i]
        answer = answers[sample_index]

        if answer:  # If answer exists
            start_char = contexts[sample_index].find(answer)
            if start_char == -1:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue
            end_char = start_char + len(answer)

            token_start_index = 0
            token_end_index = len(input_ids) - 1
            if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                for idx, (start, end) in enumerate(offsets):
                    if start <= start_char < end:
                        token_start_index = idx
                    if start < end_char <= end:
                        token_end_index = idx

            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
        else:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# Apply preprocessing to the dataset
tokenized_dataset = ds.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=ds["train"].column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)

# Load metric
metric = load("squad")

def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids

    # Convert logits to predicted start and end indices
    start_indices = predictions[:, :model.config.start_n_top].argmax(axis=1)
    end_indices = predictions[:, model.config.start_n_top:].argmax(axis=1)

    # Compute metrics
    return metric.compute(predictions=predictions, references=labels)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate(tokenized_dataset["validation"])
print(f"F1 Score: {metrics.get('f1', 'N/A')}, Exact Match (EM): {metrics.get('em', 'N/A')}")
