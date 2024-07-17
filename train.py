import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from dataset import get_train_test_splits, make_datasets

class Trainer:

    def __init__(self, model_class, hparams, err_loss_func=torch.nn.MSELoss(), device="cuda:0"):
        self.model = model_class(**hparams).to(device)
        self.hp = hparams
        self.err_loss_func = err_loss_func
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr_init"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.hp["lr_decay"])

        self.train_loss_history = []
        self.val_loss_history = []
        self.val_mse_history = []

    def n_params(self):
        return sum([len(x.view(-1)) for x in list(self.model.parameters())])

    def loss_func(self, y_true, y_pred, mask):
        n_comp = self.hp["n_comp"]
        mask = ~mask.contiguous().view(-1)
        err_loss = self.err_loss_func(y_pred.contiguous().view(-1)[mask], y_true.contiguous().view(-1)[mask])
        if isinstance(self.model, SorbNetX):
            off_diag = self.model.get_value_matrices().view(self.model.n_site, n_comp, n_comp) \
                            * (1 - torch.eye(n_comp, device=self.device).unsqueeze(0))
            reg_loss = torch.norm(off_diag, p=1) * self.hp["reg"]
        else:
            reg_loss = torch.zeros_like(err_loss, device=self.device)
        return err_loss, reg_loss

    def training_step(self, batch):
        self.opt.zero_grad()
        x, y_true = batch
        x, y_true = x.to(self.device), y_true.to(self.device)
        y_pred, attn, mask = self.model(x)
        err_loss, reg_loss = self.loss_func(y_true, y_pred, mask)
        loss = err_loss + reg_loss
        loss.backward()
        self.opt.step()
        return loss.item()

    def validation_step(self, batch):
        x, y_true = batch
        x, y_true = x.to(self.device), y_true.to(self.device)
        y_pred, attn, mask = self.model(x)
        err_loss, reg_loss = self.loss_func(y_pred, y_true, mask)
        return (err_loss + reg_loss).item(), (err_loss).item()

    def train(self, trainloader, testloader, epochs=200):
        print("Num parameters:", self.n_params())
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                running_loss = 0.0
                for i, batch in enumerate(trainloader, 0):
                    loss_value = self.training_step(batch)
                    running_loss += loss_value
                running_loss /= i + 1
                self.train_loss_history.append(running_loss)
                with torch.no_grad():
                    for batch in testloader:   
                        loss, mse = self.validation_step(batch)
                self.val_loss_history.append(loss)
                self.val_mse_history.append(mse)
                pbar.set_postfix({
                    'Training loss': running_loss,
                    'Test loss': self.val_loss_history[-1],
                    'Test MSE': self.val_mse_history[-1],
                })
                self.scheduler.step()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SorbNetX model")
    parser.add_argument("hparams", type=str,
                    help="Path to hyperparameter config JSON file")
    parser.add_argument("paths", type=str, nargs="+",
                    help="Path to dataframe files")
    parser.add_argument("-d", "--device", type=str, default="cuda:0",
                    help="Device to run training")
    parser.add_argument("-s", "--save", type=str, default=None,
                    help="Path to save model weights")
    return parser.parse_args()

def run_training(
    df_paths,
    hparams,
    batch_size=256,
    device="cuda:0",
):
    df_train, df_test, n_comp = get_train_test_splits(
        df_paths,
        hparams["train_components"],
        hparams["test_components"],
    )
    data_train, data_test = make_datasets(df_train, df_test, n_comp)
    batch_size = 256
    trainloader = DataLoader(data_train, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    testloader = DataLoader(data_test, batch_size=len(data_test), num_workers=0, shuffle=True, pin_memory=True)
    hparams["n_comp"] = n_comp
    hparams["n_state_each"] = data_train[0][0].shape[-1]
    trainer = Trainer(model_dict[hparams["model_class"]], hparams, device=device)
    print(f"Training set size: {len(data_train)}")
    print(f"Test set size: {len(data_test)}")
    trainer.train(trainloader, testloader, hparams["epoch"])
    return trainer

    

if __name__ == "__main__":
    args = parse_args()
    with open(args.hparams, "r") as f:
        hparams = json.load(f)
    device = torch.device(args.device)
    trainer = run_training(args.paths, hparams, device=device)
    if args.save:
        torch.save(trainer, args.save)


    