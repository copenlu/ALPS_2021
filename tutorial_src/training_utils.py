import torch
from typing import Dict
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
import random


def enforce_reproducibility(seed: int = 42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

def train_model(model: torch.nn.Module,
                train_dl: torch.data.utils.DataLoader,
                dev_dl: torch.data.utils.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                n_epochs: int) -> (Dict, Dict):
    loss_f = torch.nn.CrossEntropyLoss()

    best_val, best_model_weights = {'val_f1': 0}, None

    for ep in range(n_epochs):
        model.train()
        for batch in tqdm(train_dl, desc='Training'):
            optimizer.zero_grad()
            logits = model(batch[0])
            loss = loss_f(logits, batch[1])
            loss.backward()
            optimizer.step()

        val_p, val_r, val_f1, val_loss, _, _ = eval_model(model, dev_dl)
        current_val = {
            'val_p': val_p, 'val_r': val_r, 'val_f1': val_f1,
            'val_loss': val_loss, 'ep': ep
        }

        print(current_val, flush=True)

        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

        scheduler.step(val_loss)

    return best_model_weights, best_val


def eval_model(model: torch.nn.Module,
               test_dl: torch.data.utils.DataLoader,
               measure=None):
    model.eval()

    loss_f = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            logits_val = model(batch[0])
            loss_val = loss_f(logits_val, batch[1])
            losses.append(loss_val.item())

            labels_all += batch[1].detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.array(logits_all), axis=-1)

        if measure == 'acc':
            p, r = None, None
            f1 = accuracy_score(labels_all, prediction)
        else:
            p, r, f1, _ = precision_recall_fscore_support(labels_all,
                                                          prediction,
                                                          average='macro')

        print(confusion_matrix(labels_all, prediction))

    return p, r, f1, np.mean(losses), labels_all, prediction