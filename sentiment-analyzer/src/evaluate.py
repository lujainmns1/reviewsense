import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model_path, val_loader, model_class=None, tokenizer_name=None, num_labels=None, device=None):
    """
    Evaluate a trained PyTorch model.
    Args:
        model_path (str): Path to the saved model (.pt)
        val_loader (DataLoader): Validation DataLoader
        model_class (nn.Module): Model class (e.g., SentimentClassifier)
        tokenizer_name (str): tokenizer name for loading
        num_labels (int): number of classes
        device: torch.device
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = model_class(
        pretrained_name=tokenizer_name,
        num_labels=num_labels,
        dropout_rate=0.3
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Evaluation results - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    print("=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return acc, precision, recall, f1, cm
