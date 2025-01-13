from datasets import load_dataset
from huggingface_hub import login
import torch
from datasets.utils.logging import set_verbosity_info
from transformers.utils.logging import set_verbosity_info
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from accelerate import Accelerator
import os
import torch


set_verbosity_info()  # Ensure progress bars for Hugging Face datasets and transformers


login("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")
from datasets import load_dataset

print("*****************************************************************")

print("Loading the dataset...  🚀 🚀 🚀")

print("*****************************************************************")

dataset = load_dataset("Jaafer/cleaned_umls_corpus")
print("Dataset loaded successfully! 😊 😊")
print(dataset)

print("*****************************************************************")

print("Loading the tokenizert...  🚀 🚀 🚀 ")

print("*****************************************************************")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(example):
    return tokenizer(example['Text'], truncation=True, padding=True)
print("*****************************************************************")

print("Tokenizing the dataset...  🚀 🚀 🚀")

print("*****************************************************************")
    

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Inspect the tokenized dataset
print(tokenized_dataset)


if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU is available and being used.")
        use_fp16 = True
else:
        device = torch.device('cpu')
        print(" GPU is not available, using CPU.")

# Load the tokenizer and BERT model for MLM
model_checkpoint = "bert-base-uncased"

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint).to(device)

print("*****************************************************************")

print("Preparing the data collator...  🚀 🚀 🚀")

print("*****************************************************************")
    
# Prepare the data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=True, 
                                                mlm_probability=0.15)



print("Data collator prepared successfully!  😊 😊 😊 ")

print("*****************************************************************")

print("Setting up training arguments...  🚀 🚀 🚀")

print("*****************************************************************")
# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_d",
    overwrite_output_dir=True,
    num_train_epochs=10,  # Adjust as needed
    per_device_train_batch_size=16,
    save_steps=5000,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    
    logging_dir="./logs",
)

# Remove unnecessary columns from the dataset
tokenized_dataset = tokenized_dataset.remove_columns(["Text"])
tokenized_dataset
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],  # Ensure your dataset has a 'train' split
)
print("*****************************************************************")

print("Starting model training... 🚀 🚀 🚀" )

print("*****************************************************************")
# Train the model
trainer.train()




