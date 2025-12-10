import os
import hydra 
import torch
from peft import get_peft_model, LoraConfig
from training_utils.utils import training_schedule, build_dataloader


# Hydra configuration 
@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")



def main(cfg):

    # Set seed 
    torch.manual_seed(cfg.hyperparameters.seed) 

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #---------------- Get dataloaders ----------------
    # Get dataset parameters
    batch_size = cfg.dataset.batch_size
    seq_length = cfg.dataset.seq_length 

    # Get tokenizer 
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer)

    # Get datasets
    train_set = hydra.utils.instantiate(cfg.dataset.train)
    validation_set = hydra.utils.instantiate(cfg.dataset.validation)
    test_set = hydra.utils.instantiate(cfg.dataset.test)

    # Get dataloaders
    train_dataloader = build_dataloader(dataset=train_set,
                                        tokenizer=tokenizer,
                                        batch_size=batch_size,
                                        seq_length=seq_length,
                                        num_workers = 16,
                                        shuffle=True,
                                        drop_last=True,)
    
    validation_dataloader = build_dataloader(dataset=validation_set,
                                        tokenizer=tokenizer,
                                        batch_size=batch_size,
                                        seq_length=seq_length,
                                        num_workers = 16,
                                        shuffle=True,
                                        drop_last=True,)
    
    test_dataloader =  build_dataloader(dataset=test_set,
                                        tokenizer=tokenizer,
                                        batch_size=batch_size,
                                        seq_length=seq_length,
                                        num_workers = 16,
                                        shuffle=True,
                                        drop_last=True)
    

    #---------------- Get  model  ----------------
    model = hydra.utils.instantiate(cfg.model.model)


    # Apply LoRa 
    if cfg.lora.do_lora:

        lora_cfg = LoraConfig(r=cfg.lora.r,
                            lora_alpha=cfg.lora.lora_alpha,
                            lora_dropout=cfg.lora.lora_dropout,
                            bias=cfg.lora.bias,
                            target_modules=cfg.lora.target_modules,
                            task_type="CAUSAL_LM")

        model = get_peft_model(model, lora_cfg)


    # Apply method 
    model = hydra.utils.instantiate(cfg.method.apply, model).to(device)
       
    # Get optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Get the current Hydra output directory
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    os.makedirs(hydra_output_dir, exist_ok=True)
    
    # Train
    training_schedule(model, train_dataloader, validation_dataloader, optimizer, device, hydra_output_dir)

    return

if __name__ == "__main__":
    main()