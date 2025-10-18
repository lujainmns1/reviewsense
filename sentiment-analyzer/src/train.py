import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
import logging

from src.dataset import SentimentDataset
from src.model import SentimentClassifier
from src.evaluate import evaluate_model
from config.config import DATA_CONFIG, MODEL_OPTIONS, TRAINING_CONFIG
from src.utils import setup_logging, create_directories

# Setup logging
logger = setup_logging()


def select_model():
    """Allow user to select a model from MODEL_OPTIONS at runtime"""
    print("Available models:")
    for i, key in enumerate(MODEL_OPTIONS.keys()):
        print(f"{i+1}. {key} - {MODEL_OPTIONS[key]['description']}")

    while True:
        choice = input(f"Select a model (1-{len(MODEL_OPTIONS)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(MODEL_OPTIONS):
            selected_model = list(MODEL_OPTIONS.keys())[int(choice) - 1]
            model_config = MODEL_OPTIONS[selected_model]
            logger.info(f"Selected model: {selected_model}")
            return selected_model, model_config
        else:
            print("Invalid choice. Please try again.")



def load_dataset(tokenizer_name):
    dataset = SentimentDataset(
        excel_path=DATA_CONFIG["raw_path"],  # Excel file
        tokenizer_name=tokenizer_name,
        max_len=DATA_CONFIG["max_seq_length"]
    )
    return dataset



def train_model(selected_model, model_config):
    create_directories(selected_model)

    # Reproducibility
    torch.manual_seed(DATA_CONFIG["random_state"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DATA_CONFIG["random_state"])

    # Load dataset
    dataset = load_dataset(model_config["pretrained_name"])

    # Split dataset
    test_size = int(DATA_CONFIG["test_size"] * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(
        pretrained_name=model_config["pretrained_name"],
        num_labels=model_config["num_labels"],
        dropout_rate=model_config["dropout_rate"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    criterion = CrossEntropyLoss()

    for epoch in range(TRAINING_CONFIG["epochs"]):
        model.train()
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    model_save_path = f"models/trained_models/{selected_model}.pt"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Training complete. Model saved as {model_save_path}")

    # Evaluate after training dynamically
    evaluate_model(
        model_path=model_save_path,
        val_loader=val_loader,
        model_class=SentimentClassifier,
        tokenizer_name=model_config["pretrained_name"],
        num_labels=model_config["num_labels"],
        device=device
    )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    selected_model, model_config = select_model()
    train_model(selected_model, model_config)
