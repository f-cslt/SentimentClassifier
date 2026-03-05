import torch
import torch.nn as nn

def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for text, labels in train_loader:
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            # To prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_loss:.3f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return model


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text, labels in test_loader:
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text)
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total
