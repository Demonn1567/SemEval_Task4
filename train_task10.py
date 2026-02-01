import json
import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset

# --- CONFIGURATION ---
MODEL_PATH = "./nli-deberta-v3-large"
TEXT_FILE = "train_rehydrated.jsonl"
LABEL_FILE = "train_redacted.jsonl"
OUTPUT_DIR = "./final_model"

# Mac M4 Settings
MAX_LEN = 512
BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-5

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"--- RUNNING ON: {device} ---")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average='macro')}

def get_id(obj):
    if 'id' in obj: return obj['id']
    if '_id' in obj: return obj['_id']
    return None

def load_and_merge_data():
    print("Step 1: Loading Labels...")
    label_map = {}
    
    if not os.path.exists(LABEL_FILE):
        print(f"CRITICAL ERROR: {LABEL_FILE} not found.")
        exit()
        
    with open(LABEL_FILE, 'r') as f:
        for line in f:
            obj = json.loads(line)
            obj_id = get_id(obj)
            
            label = None
            if 'label' in obj: label = obj['label']
            elif 'conspiracy' in obj: label = "Yes" if obj['conspiracy'] == 1 else "No"
            
            # Logic to handle different label formats
            if label is None and 'conspiracy' in obj:
                 val = obj['conspiracy']
                 if str(val).lower() in ['1', 'yes', 'true']: label = 'Yes'
                 elif str(val).lower() in ['0', 'no', 'false']: label = 'No'

            if obj_id and label:
                label_map[obj_id] = label

    print(f"Loaded {len(label_map)} labels.")

    print("Step 2: Merging Text...")
    data = []
    with open(TEXT_FILE, 'r') as f:
        for line in f:
            obj = json.loads(line)
            obj_id = get_id(obj)
            
            if obj_id in label_map:
                obj['label'] = label_map[obj_id]
                if obj['label'] in ['Yes', 'No']:
                    data.append(obj)
            
    print(f"Successfully merged {len(data)} samples.")
    return data

def main():
    # 1. Load & Merge
    raw_data = load_and_merge_data()
    if len(raw_data) == 0:
        print("Error: No data loaded.")
        return

    df = pd.DataFrame(raw_data)
    df['label_id'] = df['label'].map({'Yes': 1, 'No': 0})
    
    # 2. Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label_id']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label_id']])
    
    # 3. Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=MAX_LEN)
    
    train_ds = train_ds.map(preprocess, batched=True)
    val_ds = val_ds.map(preprocess, batched=True)
    
    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Model
    print("Loading Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # 5. Train
    args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",  # <--- CHANGED THIS LINE (The Fix)
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=False,
        use_mps_device=True,
        save_total_limit=1,
        logging_steps=20,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving Final Model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()