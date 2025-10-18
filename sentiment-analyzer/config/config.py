"""
CONFIGURATION CENTER - SIMPLIFIED
"""

# Data configuration
DATA_CONFIG = {
    "raw_path": "data/data_Saudi_banks.xlsx",
    "max_seq_length": 128,
    "test_size": 0.2,
    "random_state": 42,
}

# Model options
MODEL_OPTIONS = {
    # 1. XLM-RoBERTa Large (State-of-the-art for multilingual)
    "xlm_roberta_large": {
        "pretrained_name": "xlm-roberta-large",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "XLM-RoBERTa Large - SOTA Multilingual",
        "parameters": "560M",
        "disk_size": "~2.1GB",
        "memory_usage": "~4-6GB",
        "size_category": "Very Large"
    },
    
    # 2. AraBERT V2 (Best for Arabic specifically)
    "bert_arabertv2": {
        "pretrained_name": "aubmindlab/bert-base-arabertv2",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "AraBERT V2 - Best Arabic BERT",
        "parameters": "135M",
        "disk_size": "~480MB",
        "memory_usage": "~1-2GB",
        "size_category": "Base"
    },
    
    # 3. MARBERT V2 (Top for Modern Standard Arabic)
    "marbertv2": {
        "pretrained_name": "UBC-NLP/MARBERTv2",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "MARBERT V2 - Modern Standard Arabic",
        "parameters": "135M",
        "disk_size": "~480MB",
        "memory_usage": "~1-2GB",
        "size_category": "Base"
    },
    
    # 4. Arabic BERT Large (Larger capacity)
    "arabic_bert_large": {
        "pretrained_name": "asafaya/bert-large-arabic",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "Arabic BERT Large - High Capacity",
        "parameters": "335M",
        "disk_size": "~1.2GB",
        "memory_usage": "~3-4GB",
        "size_category": "Large"
    },
    
    # 5. CAMeL-BERT MSA (From prestigious CAMeL Lab)
    "camelbert_msa": {
        "pretrained_name": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "CAMeL-BERT MSA - Research Grade",
        "parameters": "135M",
        "disk_size": "~480MB",
        "memory_usage": "~1-2GB",
        "size_category": "Base"
    },
    
    # 6. Multilingual BERT (Classic, widely used)
    "multilingual_bert": {
        "pretrained_name": "bert-base-multilingual-cased",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "Multilingual BERT - Industry Standard",
        "parameters": "177M",
        "disk_size": "~690MB",
        "memory_usage": "~1.5-2.5GB",
        "size_category": "Base+"
    },
    
    # 7. AraBERT V02 Twitter (Best for social media)
    "bert_arabertv02_twitter": {
        "pretrained_name": "aubmindlab/bert-base-arabertv02-twitter",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "AraBERT Twitter - Social Media Optimized",
        "parameters": "135M",
        "disk_size": "~480MB",
        "memory_usage": "~1-2GB",
        "size_category": "Base"
    },
    
    # 8. XLM-RoBERTa Base (Balanced performance/speed)
    "xlm_roberta_base": {
        "pretrained_name": "xlm-roberta-base",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "XLM-RoBERTa Base - Best Balance",
        "parameters": "278M",
        "disk_size": "~1.1GB",
        "memory_usage": "~2-3GB",
        "size_category": "Base+"
    },
    
    # 9. Arabic ELECTRA (Efficient architecture)
    "arabic_electra": {
        "pretrained_name": "moha/arabert-electra",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "Arabic ELECTRA - Efficient & Fast",
        "parameters": "135M",
        "disk_size": "~480MB",
        "memory_usage": "~1-2GB",
        "size_category": "Base"
    },
    
    # 10. QARiB BERT (Qatar's high-quality model)
    "qarib_bert": {
        "pretrained_name": "qarib/bert-base-qarib",
        "num_labels": 3,
        "dropout_rate": 0.3,
        "description": "QARiB BERT - Qatar Arabic Resources",
        "parameters": "135M",
        "disk_size": "~480MB",
        "memory_usage": "~1-2GB",
        "size_category": "Base"
    }
}


# Select model dynamically
SELECTED_MODEL = "bert_arabertv2"
MODEL_CONFIG = MODEL_OPTIONS[SELECTED_MODEL]

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "save_plots": True,
    "plot_dir": "logs/plots/"
}

# Logging configuration
LOGGING_CONFIG = {
    "log_dir": "logs/",
    "level": "INFO"
}
