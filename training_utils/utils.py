import os
import json 
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


# --------------Dataset processing functions--------------
def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }

def preprocess_data_hf(dataset, tokenizer, seq_length = None,  group_size=8, num_proc=4):
    """
    Memory-efficient preprocessing for BLT.
    Args:
        dataset: Hugging Face dataset with "text" column
        tokenizer: BLT tokenizer
        max_length: int, tokenizer.model_max_length if None
        group_size: int, BLT group size (default 8)
        pad_token_id: int, default tokenizer.pad_token_id
        num_proc: int, parallel processes for map
    Returns:
        tokenized and chunked Hugging Face dataset
    """
    # Get max length and pad token 
    max_length = tokenizer.model_max_length
    pad_token_id = tokenizer.pad_token_id

    # Seq length needs to be < max length and a multiple of group size
    if (seq_length is None) or (seq_length > max_length) or (seq_length % group_size != 0):
        seq_length = max_length

    # Tokenize each sample individually
    def tokenize_fn(example):
        tokens = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        return {"input_ids": tokens}

    tokenized_data = dataset.map(tokenize_fn, batched=False, num_proc=num_proc)

    # Flatten all tokens into chunks of seq_length
    def chunk_fn(batch):

        # Flatten list of tokens 
        all_tokens = sum(batch["input_ids"], [])

        # Pad all_tokens to be a multiple of seq_length
        rest = len(all_tokens) % seq_length
        if rest != 0 :
            all_tokens += [pad_token_id] * (seq_length - rest)

        # List of sequences
        sequences = [all_tokens[i:i+seq_length] for i in range(0, len(all_tokens), seq_length)]

        return {"input_ids": sequences}

    chunked_data = tokenized_data.map(chunk_fn, batched=True, batch_size=1000, remove_columns=tokenized_data.column_names)
    
    return HFTokenizedDataset(chunked_data, pad_token_id)

class HFTokenizedDataset(Dataset):
    def __init__(self, hf_dataset, pad_token_id):
        self.dataset = hf_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataset[idx]["input_ids"], dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
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