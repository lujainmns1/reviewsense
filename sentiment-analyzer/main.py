from src.train import select_model, train_model

if __name__ == "__main__":
    # Let user choose the model at runtime
    selected_model, model_config = select_model()
    
    # Train the selected model
    train_model(selected_model, model_config)
