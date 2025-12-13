import torch
import torch.nn as nn
from tqdm import tqdm

from config import Training_config
from data import Create_DataLoader
from Transformer import Transformer


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train model for a single epoch

    Args:
        model : The transformer model
        train_loader : DataLoader for training data
        optimizer : Optimizer (AdamW, etc.)
        criterion : Loss Function
        device : 'cuda' or 'cpu'
        epoch : Current epoch number

    Returns:
        Average loss for epoch
    """

    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    # initialize a progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (input_ids, output_ids) in enumerate(pbar):
        # Loading the data into the device
        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids)

        loss = criterion(
            logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
            output_ids.view(-1),  # (batch*seq_len)
        )

        # Backward pass
        loss.backward()

        # Clip gradients (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Updating weights
        optimizer.step()

        # Tracking total loss
        total_loss += loss.item()

        # Updating the progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    average_loss = total_loss / num_batches
    return average_loss


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for input_ids, target_ids in tqdm(val_loader, desc="Evaluating"):
            # Transferring data to the device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Performing forward pass
            logits = model(input_ids)

            # Calculating the loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Saving the model checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, filepath)
    print("-Checkpoint saved...")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded | epoch : {epoch} | loss : {loss}")
    return epoch, loss


def main():
    config = Training_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load and split data - FIXED: Added train/val split
    with open("Training_data.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # Split into paragraphs or lines
    text_segments = [seg.strip() for seg in content.split("\n\n") if seg.strip()]

    # Split 80% train, 20% validation
    split_idx = int(len(text_segments) * 0.8)
    train_texts = text_segments[:split_idx]
    val_texts = text_segments[split_idx:]

    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")

    model = Transformer(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout_rate=config.dropout_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-Total Model parameters : {total_params}")
    print(f"-Trainable Parameters : {trainable_params}")
    print(f"-Non Trainable parameters : {total_params - trainable_params}")

    # Memory estimation
    param_size_mb = total_params * 4 / (1024**2)
    print(f"-Model Size (FP32) : {param_size_mb:.2f} MB")

    print("Creating Data Loaders...")
    train_loader, val_loader, tokenizer = Create_DataLoader(
        train_texts=train_texts,
        val_texts=val_texts,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        stride=config.stride,
    )

    # Setup training - FIXED: Use tokenizer.pad_token_id instead of 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        print("=" * 100)
        print(f"epoch : {epoch}/{config.num_epochs}")
        print("." * 50)

        training_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        print(f"Training loss : {training_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Validation loss : {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate : {current_lr}")

        # Save the checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                filepath=f"{config.checkpoint_dir}/best_model.pt",
            )

        # Save Regular checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                filepath=f"{config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt",
            )

    print("=" * 100)
    print("Training complete...")


if __name__ == "__main__":
    main()
