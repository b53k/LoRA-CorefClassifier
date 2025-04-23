import os
import tqdm
import warnings
import re

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=re.escape("You are using `torch.load` with `weights_only=False`")
)

import torch
import random
import argparse
import logging
import warnings
import numpy as np

from datasets import load_dataset
from dataset import MentionPairDataset
from utility import *
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


#warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_validation(model, classifier, dataloader, tokenizer, DEVICE, score, max_batches=10):
    model.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    m_id = tokenizer.convert_tokens_to_ids('<m>')
    m_end_id = tokenizer.convert_tokens_to_ids('</m>')

    val_subset = random.sample(list(dataloader), min(max_batches, len(dataloader)))

    with torch.no_grad():
        for batch_x, batch_y in val_subset:
            tokens = tokenizer(batch_x, padding="max_length", max_length=512, return_tensors="pt", truncation=True, add_special_tokens=True)
            input_ids = tokens['input_ids'].to(DEVICE)
            attention_mask = tokens['attention_mask'].to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            try:
                outputs = model(input_ids, attention_mask)
                hidden_states = outputs['last_hidden_state']
                reps = extract_mention_repr(input_ids, hidden_states, m_id, m_end_id, include_scores=score)
            except ValueError as e:
                # Sometimes truncation clips the second mention.
                continue

            logits = classifier(reps).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).int()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    return precision, recall, f1, acc


# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Learning rate")
    parser.add_argument("--score", action="store_true", help="Whether to include doc-level score tokens")
    parser.add_argument('--save_dir', type=str, default="models", help="Directory to save model")
    parser.add_argument('--load_checkpoint', action='store_true', help="Load saved checkpoint if exists")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/training_log_{'scored' if args.score else 'plain'}.txt"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(0)

    tokenizer = AutoTokenizer.from_pretrained('data/tokenizer_train_scored' if args.score else 'data/tokenizer_train')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(len(tokenizer))

    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=['query', 'values'],
        lora_dropout=0.1,
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model = get_peft_model(model, config).to(DEVICE)
    print ('-------------------------')
    model.print_trainable_parameters()
    print ('-------------------------')
    print ('')
    print ('Training doc_quality aware model' if args.score else 'Training baseline model')
    print ('')

    if not args.score:
        classifier = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size * 4, model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(model.config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to(DEVICE)
    else:
        classifier = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size * 8, model.config.hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(model.config.hidden_size * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to(DEVICE)

    # Loss Function and Optimizer
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr = args.lr)

    # Load datasets
    train_data = torch.load('data/mention_pairs_train_scored.pt' if args.score else 'data/mention_pairs_train.pt')
    val_data = torch.load('data/mention_pairs_validation_scored.pt' if args.score else 'data/mention_pairs_validation.pt')
    train_loader = DataLoader(train_data, batch_size=2, num_workers=0, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=2, num_workers=0, shuffle=True, pin_memory=True)

    m_id = tokenizer.convert_tokens_to_ids('<m>')
    m_end_id = tokenizer.convert_tokens_to_ids('</m>')

    save_path = os.path.join(args.save_dir, 'model_scored.pt' if args.score else 'model.pt')

    # Load Checkpoint if available
    if args.load_checkpoint and os.path.exists(save_path):
        torch.cuda.empty_cache()
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"[INFO] Loaded model and optimizer from {save_path}")
        logging.info(f'Checkpoint loaded from {save_path}')


    best_val_f1 = 0
    patience = 4
    trigger_times = 0
    val_check_batch = 250

    for epoch in tqdm.trange(args.epochs, desc="Training", unit="epoch"):
        with tqdm.tqdm(train_loader, desc=f'epoch {epoch+1}', unit='batch', total=len(train_loader), position=0, leave=True) as batch_iterator:
            model.train()
            classifier.train()
            total_loss = 0.0

            for i, (batch_x, batch_y) in enumerate(batch_iterator, start=1):
                tokens = tokenizer(batch_x, padding="max_length", max_length=512, return_tensors="pt", truncation=True, add_special_tokens=True)
                input_ids = tokens['input_ids'].to(DEVICE)
                attention_mask = tokens['attention_mask'].to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()

                try:
                    outputs = model(input_ids, attention_mask)
                    hidden_states = outputs['last_hidden_state']            # [batch_size, token_len, hidden_dim] : [4, 512, 768]
                    reps = extract_mention_repr(input_ids, hidden_states, m_id, m_end_id, include_scores=args.score)
                except ValueError as e:
                    #tqdm.tqdm.write(f"[Skipping example {i}] Reason: {e}")
                    continue

                logits = classifier(reps).squeeze(-1)
                loss = loss_fn(logits, batch_y)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                batch_iterator.set_postfix(
                    {
                        'Mean Loss': total_loss/i,
                        'Current Loss': loss.item()
                    }
                )

                if (i % val_check_batch == 0) or (i == len(train_loader)):
                    precision, recall, f1, acc = run_validation(model, classifier, val_loader, tokenizer, DEVICE, score=args.score, max_batches=50)
                    print (f'[Validation]--- Precision: {precision:.4f} || Recall: {recall:.4f} || F1: {f1:.4f} || Accuracy: {acc:.4f}')
                    logging.info(f"Epoch {epoch+1}, Batch {i} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")
                    print ('')

                    # Reset mode back to training
                    model.train()
                    classifier.train()

                    if f1 > best_val_f1 + 0.001:
                        best_val_f1 = f1
                        trigger_times = 0
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'classifier_state_dict': classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
                        logging.info(f"Model saved at {save_path} with F1: {f1:.4f}")
                    else:
                        trigger_times += 1
                        if trigger_times >= patience:
                            print(f'[Early Stop] No improvement for {patience * val_check_batch} batches.')
                            logging.info("Training stopped early due to plateau.")
                            return

    logging.info(f'Training completed!')
# ---------------------------------------------------

if __name__=="__main__":
    main()
