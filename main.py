from model import LSTMClassifier
from dataset import get_dataloaders
from train import train_model

def main():
    train_loader, test_loader, vocab = get_dataloaders(batch_size=64)

    vocab_size = len(vocab)
    model = LSTMClassifier(vocab_size=vocab_size, pad_idx=vocab.pad_idx)

    print("Training model...")
    train_model(model, train_loader, test_loader, epochs=30)

if __name__ == "__main__":
    main()
