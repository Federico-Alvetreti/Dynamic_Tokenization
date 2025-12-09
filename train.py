import os
import hydra 
import torch
from peft import get_peft_model, LoraConfig
from training_utils.utils import training_schedule, preprocess_data_hf, collate_fn

# Hydra configuration 
@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")



def main(cfg):

    # Set seed 
    torch.manual_seed(cfg.hyperparameters.seed) 

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataset parameters
    batch_size = cfg.dataset.batch_size
    seq_length = cfg.dataset.seq_length 

    # Get dataset
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    val_dataset = hydra.utils.instantiate(cfg.dataset.test)

    # Get model & tokenizer
    model = hydra.utils.instantiate(cfg.model.model)
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer)
    
    # Apply LoRa
    if cfg.lora.do_lora:

        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM")

        model = get_peft_model(model, lora_cfg)

    
    # Add importance 
    model = hydra.utils.instantiate(cfg.method.apply, model).to(device)

    for _, p in model.model.named_parameters():
        p.requires_grad = False


    # Tokenize dataset
    train_tokenized = preprocess_data_hf(train_dataset, tokenizer, seq_length)
    val_tokenized = preprocess_data_hf(val_dataset, tokenizer, seq_length)

    # Get dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_tokenized, shuffle=True, drop_last=True, batch_size=batch_size, num_workers = 16, collate_fn = collate_fn)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_tokenized, shuffle=False, batch_size=batch_size, num_workers = 16, collate_fn = collate_fn)

    # Get optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Get the current Hydra output directory
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    os.makedirs(hydra_output_dir, exist_ok=True)
    
    # Train
    training_schedule(model, train_dataloader, val_dataloader, optimizer, device, hydra_output_dir)

    return

if __name__ == "__main__":
    main()