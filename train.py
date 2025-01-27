import os
import torch
import torch.nn as nn
from model import *

def train(
    model, 
    train_loader, 
    val_loader=None, 
    epochs=50, 
    lr=0.0005, 
    patience=5, 
    dataset_name="dataset",  # Pass the dataset name here
    save_best_model=False, 
    best_model_path="best_model.pth"
):
    log_file = f"{dataset_name}_training_log.txt"  # Dynamic log file name
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Open the log file for writing
    with open(log_file, "w") as log:
        log.write("Epoch\tTrain_Loss\tTrain_Accuracy\tVal_Loss\tVal_Accuracy\n")

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss, correct_preds, total = 0, 0, 0
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data)

                # Use mask if available
                if hasattr(data, 'train_mask'):
                    mask = data.train_mask
                    loss = criterion(out[mask], data.y[mask])
                    _, predicted = out[mask].max(dim=1)
                    correct_preds += predicted.eq(data.y[mask]).sum().item()
                    total += mask.sum().item()
                else:
                    loss = criterion(out, data.y)
                    _, predicted = out.max(dim=1)
                    correct_preds += predicted.eq(data.y).sum().item()
                    total += data.y.size(0)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_accuracy = correct_preds / total
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%")

            # Validation phase
            if val_loader:
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for data in val_loader:
                        out = model(data)
                        if hasattr(data, 'val_mask'):
                            mask = data.val_mask
                            loss = criterion(out[mask], data.y[mask])
                            _, predicted = out[mask].max(dim=1)
                            val_correct += predicted.eq(data.y[mask]).sum().item()
                            val_total += mask.sum().item()
                        else:
                            loss = criterion(out, data.y)
                            _, predicted = out.max(dim=1)
                            val_correct += predicted.eq(data.y).sum().item()
                            val_total += data.y.size(0)
                        val_loss += loss.item()
                val_accuracy = val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

                # Save the best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                    if save_best_model:
                        torch.save(model.state_dict(), best_model_path)
                        print(f"Best model saved to {best_model_path}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping triggered.")
                        break

                # Log results
                log.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{train_accuracy*100:.2f}\t{avg_val_loss:.4f}\t{val_accuracy*100:.2f}\n")
            else:
                log.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{train_accuracy*100:.2f}\n")


def evaluate(model, test_loader, dataset_name="dataset", save_results=True):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_loader:
            out = model(data)

            # Use mask if available
            if hasattr(data, 'test_mask'):
                mask = data.test_mask
                loss = criterion(out[mask], data.y[mask])
                _, predicted = out[mask].max(dim=1)
                correct += predicted.eq(data.y[mask]).sum().item()
                total += mask.sum().item()
            else:
                loss = criterion(out, data.y)
                _, predicted = out.max(dim=1)
                correct += predicted.eq(data.y).sum().item()
                total += data.y.size(0)
            test_loss += loss.item()

    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    # Save test results
    if save_results:
        log_file = f"{dataset_name}_test_results.txt"
        with open(log_file, "w") as log:
            log.write(f"Test Loss: {avg_loss:.4f}\nTest Accuracy: {accuracy * 100:.2f}%\n")
        print(f"Test results saved to {log_file}")

    return avg_loss, accuracy
