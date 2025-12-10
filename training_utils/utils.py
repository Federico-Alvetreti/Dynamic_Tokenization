import os
import json 
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# --------------Dataset processing functions--------------

# Collate function to batch 
def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }

# Tokenize a whole text dataset
def tokenize_dataset(dataset, tokenizer):
    print(f"Tokenizing Dataset")
    def tokenize_fn(x):
        return {"input_ids": tokenizer(x["text"], add_special_tokens=False)["input_ids"]}

    tokenized_dataset = dataset.map(
                tokenize_fn,
                batched=True,
                batch_size=10000)
    
    return tokenized_dataset

# Chunk each batch to a fixed sequence length 
def chunk_tokenized_dataset(tokenized_dataset, seq_length, pad_token_id):

    print(f"Chunking dataset in sequences of length {seq_length}")
    def chunk_fn(batch):

        all_tokens = sum(batch["input_ids"], [])
        remainder = len(all_tokens) % seq_length

        if remainder != 0:
            all_tokens += [pad_token_id] * (seq_length - remainder)

        sequences = [all_tokens[i : i + seq_length] for i in range(0, len(all_tokens), seq_length)]

        return {"input_ids": sequences}

    chunked_dataset = tokenized_dataset.map(
        chunk_fn,
        batched=True,
        batch_size=50,
        remove_columns=tokenized_dataset.column_names)
    
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
    drop_last=False,
):

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
        collate_fn=collate_fn)

    return dataloader

# --------------------------------------------------------


# Standard training phase 
def training_phase(model, train_data_loader, optimizer, device, plot):

    if plot:
        print("\nTraining phase:")

    # Initialize train loss 
    train_loss = 0.0

    # Set the model in training mode
    model.train()
    
    # Forward the train set
    for batch in tqdm(train_data_loader, disable=not plot):

        # Get input ids and attention mask
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None).to(device)
        
        # Forward 
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        
        # Get batch loss
        loss = outputs.loss +  model.get_loss()
        train_loss += loss.detach().cpu().item()

        # print("importance loss:", model.get_loss().detach().cpu().item())
        # print("language modeling loss:", outputs.loss.detach().cpu().item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = train_loss / len(train_data_loader)
    
    return avg_train_loss

# Standard validation phase
def validation_phase(model, val_data_loader, device, plot):
    if plot:
        print("\nValidation phase:")

    # Initialize val  loss 
    val_loss = 0.0

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

            # Get batch loss
            loss = outputs.loss
            val_loss += loss.cpu().item()

    avg_val_loss = val_loss / len(val_data_loader)
    return avg_val_loss

# Standard training schedule
def training_schedule(model,
                      train_data_loader,
                      val_data_loader,
                      optimizer,
                      device,
                      hydra_output_dir,
                      patience=5,
                      max_epochs=1000,
                      plot=True,
                      save_model=True):

    # Init Results
    train_losses, val_losses = [], []
    results_file = os.path.join(hydra_output_dir, "training_results.json")

    # Set patience 
    patience_counter = 0
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs+1):
        torch.cuda.empty_cache()
        if plot:
            print(f"\n\nEPOCH {epoch}")

        # Get losses
        avg_train_loss = training_phase(model, train_data_loader, optimizer, device, plot)
        avg_val_loss = validation_phase(model, val_data_loader, device, plot)

        # Store them
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if plot:
            print(f"\nTrain loss: {avg_train_loss:.4f}; Val loss: {avg_val_loss:.4f}")

        # Save results
        results = {"Train losses": train_losses, "Val losses": val_losses}
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