import os
import json 
import torch
import numpy as np 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import chain

# --------------DATA PROCESSING --------------

# Collate function to batch 
def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# ------------------ Optimized tokenization ------------------
def tokenize_dataset(dataset, tokenizer, batch_size=10000, num_proc=8):
    print("Tokenizing Dataset...")

    example = dataset[0]
    possible_keys = ["text", "Text", "content", "sentence"]

    for k in possible_keys:
        if k in example:
            text_key = k
            break
    else:
        raise KeyError(f"No text key found in batch: {batch.keys()}")


    def tokenize_fn(batch):
        texts = batch[text_key]
        input_ids = [
            torch.tensor(tokenizer(t, add_special_tokens=False)["input_ids"], dtype=torch.long)
            for t in texts
        ]
        return {"input_ids": input_ids}

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    return tokenized_dataset

# ------------------ Optimized chunking ------------------
def chunk_tokenized_dataset(tokenized_dataset, seq_length, pad_token_id, batch_size=1000, num_proc=8):
    print(f"Chunking dataset in sequences of length {seq_length}...")

    def chunk_fn(batch):
        # Flatten the list of tensors efficiently
        all_tokens = np.concatenate([np.array(ids, dtype=np.int32) for ids in batch["input_ids"]])
        
        # Pad to multiple of seq_length
        remainder = len(all_tokens) % seq_length
        if remainder != 0:
            all_tokens = np.pad(all_tokens, (0, seq_length - remainder), constant_values=pad_token_id)
        
        # Reshape into (num_sequences, seq_length)
        sequences = all_tokens.reshape(-1, seq_length)

        # Return as list of lists for datasets compatibility
        return {"input_ids": sequences.tolist()}

    chunked_dataset = tokenized_dataset.map(
        chunk_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=tokenized_dataset.column_names,
    )

    return chunked_dataset

# Standard Dataset that return input_ids and attention mask 
class standard_dataset(Dataset):

    def __init__(self, chunked_tokenized_dataset, pad_token_id):
        self.dataset = chunked_tokenized_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Return input_ids and attention_mask 
        input_ids = torch.tensor(self.dataset[idx]["input_ids"], dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {"input_ids": input_ids,
                "attention_mask": attention_mask}

# Final function to build dataloader
def build_dataloader(
    dataset,
    tokenizer,
    batch_size,
    seq_length=None,
    group_size=8,
    num_workers=16,
    shuffle=True,
    drop_last=False,):

    max_length = tokenizer.model_max_length
    pad_token_id = tokenizer.pad_token_id

    # Validate seq_length
    if (seq_length is None
        or seq_length > max_length
        or seq_length % group_size != 0
    ):
        seq_length = max_length

    # Tokenization
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Chunking
    chunked_tokenized_dataset = chunk_tokenized_dataset(tokenized_dataset, seq_length, pad_token_id)

    # Build Dataset
    torch_dataset = standard_dataset(chunked_tokenized_dataset, pad_token_id)

    # Build dataloader 
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn)

    return dataloader

# -------------TRAINING SCHEDULE / EVALUATION --------------------

# Standard training phase 
def training_phase(model, train_data_loader, optimizer, device, plot):

    if plot:
        print("\nTraining phase:")

    # Initialize train loss 
    train_task_loss = 0.0
    train_importance_loss = 0.0

    # Set the model in training mode
    model.train()
    
    # Forward the train set
    for batch in tqdm(train_data_loader, disable=not plot):

        # Get input ids and attention mask
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask", None).to(device,non_blocking=True)
        
        # Forward 
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        

        # Get full loss 
        loss = outputs.loss +  model.get_loss()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Store single losses
        train_task_loss += outputs.loss.detach().cpu().item()
        train_importance_loss += model.get_loss().detach().cpu().item()


    # Average losses 
    avg_train_task_loss = train_task_loss / len(train_data_loader)
    avg_train_importance_loss = train_importance_loss / len(train_data_loader)
    
    return avg_train_task_loss, avg_train_importance_loss

# Standard validation phase
def validation_phase(model, val_data_loader, device, plot):
    if plot:
        print("\nValidation phase:")

    # Initialize val loss 
    val_task_loss = 0.0
    val_importance_loss = 0.0

    # Set the model in eval mode
    model.eval()

    # Forward the validation set
    for batch in tqdm(val_data_loader, disable=not plot):

        # Get input ids and attention mask
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None).to(device)

        # Forward 
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Store single losses
            val_task_loss += outputs.loss.detach().cpu().item()
            val_importance_loss += model.get_loss().detach().cpu().item()

    # Average losses 
    avg_val_task_loss = val_task_loss / len(val_data_loader)
    avg_val_importance_loss = val_importance_loss / len(val_data_loader)

    return avg_val_task_loss, avg_val_importance_loss

# Standard training schedule
def training_schedule(model,
                      train_data_loader,
                      val_data_loader,
                      optimizer,
                      device,
                      hydra_output_dir,
                      patience=5,
                      max_epochs=50,
                      plot=True,
                      save_model=True):

    # Init Results
    train_task_losses, train_importance_losses, val_task_losses, val_importance_losses = [], [], [], []
    results_file = os.path.join(hydra_output_dir, "training_results.json")

    # Set patience 
    patience_counter = 0
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs+1):
        torch.cuda.empty_cache()
        if plot:
            print(f"\n\nEPOCH {epoch}")

        # Get losses
        avg_train_task_loss, avg_train_importance_loss = training_phase(model, train_data_loader, optimizer, device, plot)
        avg_val_task_loss, avg_val_importance_loss = validation_phase(model, val_data_loader, device, plot)

        # Get the avg_val_loss 
        avg_val_loss = avg_val_task_loss + avg_val_importance_loss

        # Store them
        train_task_losses.append(avg_train_task_loss)
        train_importance_losses.append(avg_train_importance_loss)
        val_task_losses.append(avg_val_task_loss)
        val_importance_losses.append(avg_val_importance_loss)
        

        # Save results
        results = {"Train task losses": train_task_losses,
                    "Train importance losses": train_importance_losses,
                    "Val task losses": val_task_losses,
                    "val importance losses": val_importance_losses}

        os.makedirs(hydra_output_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        # Patience + save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_model:
                model_file = os.path.join(hydra_output_dir, "model.pt")
                torch.save(model.state_dict(), model_file)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if plot:
                print(f"Stopping early at epoch {epoch} due to patience.")
            break


# def evaluate(model, tokenizer, test_dataloader, device, hydra_output_dir, plot = False)

#     # Test evaluation 
#     avg_test_task_loss, avg_test_importance_loss = validation_phase(model, test_dataloader, device, plot)

#     # Tokenization quality 
#     #
#     # Now here i would like to take a few test samples (first 10?) and see the tokenization !
#     #
#     #
#     #
