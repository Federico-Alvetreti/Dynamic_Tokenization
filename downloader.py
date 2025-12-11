import os
import datasets
import transformers

# Folders paths
storage_path = "/leonardo_scratch/fast/IscrC_ELLIF-HE/federicoalvetreti"
data_dir = storage_path + "/data"
models_dir = storage_path + "/models"

# ---- DOWNLOAD DATASETS
datasets.load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    cache_dir=data_dir)

datasets.load_dataset(
    "roneneldan/TinyStories",
    cache_dir=data_dir)

datasets.load_dataset(
    "Trelis/tiny-shakespeare",
    cache_dir=data_dir)

# ---- DOWNLOAD MODELS
transformers.AutoModelForCausalLM.from_pretrained(
    "itazap/blt-1b-hf",
    cache_dir=models_dir)

# ---- DOWNLOAD TOKENIZERS
transformers.AutoTokenizer.from_pretrained(
    "itazap/blt-1b-hf",
    cache_dir=models_dir)

