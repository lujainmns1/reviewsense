import os
import torch
from src.model import SentimentClassifier
from src.dataset import SentimentDataset
from config.config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Inference:
    def __init__(self, model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "models/trained_models/"
        
        # Load latest model if none provided
        if model_name is None:
            models = self.list_models()
            if not models:
                raise FileNotFoundError("No trained models found in models/trained_models/")
            model_name = models[-1]
        
        self.model_path = os.path.join(self.model_dir, model_name)
        print(f"Loading model: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["pretrained_name"])
        self.model = SentimentClassifier(
            pretrained_name=MODEL_CONFIG["pretrained_name"],
            num_labels=MODEL_CONFIG["num_labels"],
            dropout_rate=MODEL_CONFIG["dropout_rate"]
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def list_models(self):
        files = os.listdir(self.model_dir)
        return sorted([f for f in files if f.endswith(".pt")])

    def choose_model_interactive(self):
        models = self.list_models()
        if not models:
            raise FileNotFoundError("No trained models found in models/trained_models/")
        
        print("Available models:")
        for idx, m in enumerate(models):
            print(f"{idx+1}. {m}")
        
        choice = input(f"Choose a model (1-{len(models)}): ")
        try:
            choice_idx = int(choice) - 1
            if choice_idx not in range(len(models)):
                raise ValueError
            self.model_path = os.path.join(self.model_dir, models[choice_idx])
            print(f"Selected model: {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        except ValueError:
            print("Invalid choice. Using latest model.")

    def predict(self, texts, max_len=DATA_CONFIG["max_seq_length"]):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu().numpy()

    def evaluate_on_test(self):
        import pandas as pd
        df = pd.read_excel(DATA_CONFIG["raw_path"])
        texts = df["Tweet"].tolist()
        labels = [LABEL_MAP[label] for label in df["Final annotation"]]

        dataset = SentimentDataset(
            excel_path=DATA_CONFIG["raw_path"],
            tokenizer_name=MODEL_CONFIG["pretrained_name"],
            max_len=DATA_CONFIG["max_seq_length"]
        )

        test_size = int(DATA_CONFIG["test_size"] * len(dataset))
        train_size = len(dataset) - test_size
        _, test_dataset = random_split(dataset, [train_size, test_size])

        test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG["batch_size"])

        all_preds, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )

        print("\n=== Test Dataset Evaluation ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("==============================\n")



# Map string labels to numbers 
LABEL_MAP = {"NEG": 0, "NEU": 1, "POS": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

if __name__ == "__main__":
    # Initialize Inference
    inf = Inference()
    
    # Choose model interactively (optional)
    inf.choose_model_interactive()
    
    # Evaluate on test dataset (uses proper test split)
    inf.evaluate_on_test()

    # Realistic example inference (10 comments)
    examples = [
        "الخدمة في هذا البنك ممتازة وسريعة",
        "انتظرت أكثر من ساعة دون أن يرد علي أحد",
        "التجربة كانت عادية لا شيء مميز",
        "الموظفون متعاونون ولطيفون جدًا",
        "لا أنصح بهذا البنك، تعاملهم سيء",
        "استلمت تمويلي بسرعة وسهولة",
        "الموقع الإلكتروني معقد ويصعب استخدامه",
        "أحببت طريقة التعامل مع العملاء هنا",
        "الرسوم عالية جدًا مقارنة بالخدمات",
        "المعاملة كانت متوسطة، لا جيدة ولا سيئة"
    ]

    # True labels for these 10 comments (for example)
    true_labels = ["POS", "NEG", "NEU", "POS", "NEG", "POS", "NEG", "POS", "NEG", "NEU"]
    true_labels_num = [LABEL_MAP[label] for label in true_labels]

    # Predict
    preds_num = inf.predict(examples)

    # Print results neatly
    print("\n=== Inference Results ===")
    for text, true, pred in zip(examples, true_labels_num, preds_num):
        print(f"Text: {text}\n  True: {INV_LABEL_MAP[true]}, Predicted: {INV_LABEL_MAP[pred]}\n")

    # Calculate simple accuracy
    acc = accuracy_score(true_labels_num, preds_num)
    print(f"Accuracy on 10 examples: {acc:.2f}")
