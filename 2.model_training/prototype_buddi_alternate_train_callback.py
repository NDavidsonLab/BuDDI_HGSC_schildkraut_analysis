import tensorflow as tf
import pandas as pd

class AlternatingTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, unsupervised_model, unsupervised_data, epochs_per_alt=1):
        self.unsupervised_model = unsupervised_model
        self.unsupervised_data = unsupervised_data
        self.epochs_per_alt = epochs_per_alt
        self.unsupervised_logs = []  # Store loss history

    def on_epoch_end(self, epoch, logs=None):
        """Train the unsupervised model after each supervised epoch."""
        print(f"\nSwitching to unsupervised training (Epoch {epoch+1})...")
        history = self.unsupervised_model.fit(
            self.unsupervised_data, 
            epochs=self.epochs_per_alt, 
            verbose=1
        )
        
        # Store unsupervised loss
        unsup_loss = history.history  # Dictionary of all losses
        self.unsupervised_logs.append({"epoch": epoch+1, **unsup_loss})
        print(f"Unsupervised Training Loss: {unsup_loss}")

    def on_train_end(self, logs=None):
        """Save unsupervised training logs at the end."""
        df = pd.DataFrame(self.unsupervised_logs)
        df.to_csv("unsupervised_training_log.csv", index=False)
        print("Unsupervised training log saved!")