import matplotlib.pyplot as plt

from model import *
from train import Trainer
from sklearn.metrics import r2_score
from dataset import get_train_test_splits, make_datasets


paths = ["data-full/MFI-0.csv", "data-binary-ternary/MFI-0.csv"]

model_file = "sorbnetx-2.pt"

trainer = torch.load(model_file)
trainer.model = trainer.model.to("cpu")

df_train, df_test, n_comp = get_train_test_splits(
    paths,
    trainer.hp["train_components"],
    trainer.hp["test_components"],
)
data_train, data_test = make_datasets(df_train, df_test, n_comp)
x_train, y_train = data_train.tensors
x_test, y_test = data_test.tensors

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(trainer.train_loss_history)
ax.plot(trainer.val_loss_history)
ax.legend(["Training loss", "Test loss"])
ax.set_xlabel("Epochs")
ax.set_yscale("log")
plt.savefig(f"{model_file.rstrip('.pt')}_loss.png", format="png", bbox_inches="tight")


molecules = ['pX', 'oX', 'mX', 'bz', 'tol', 'eb', 'H2', 'eth']
with torch.no_grad():
    y_train_pred, _, _ = trainer.model(x_train)
    y_test_pred, _, _ = trainer.model(x_test)
    y_train_pred[torch.isnan(y_train_pred)] = 0
    y_test_pred[torch.isnan(y_test_pred)] = 0

for i in range(n_comp):
    print(molecules[i], r2_score(y_train[:, i], y_train_pred[:, i]))
for i in range(n_comp):
    print(molecules[i], r2_score(y_test[:, i], y_test_pred[:, i]))

fig, ax = plt.subplots(figsize=(4, 3))
for i in range(n_comp):
    ax.scatter(y_train[:, i], y_train_pred[:, i], s=1)
ax.legend(molecules)
ax.plot([0, 1], [0, 1], ls='--', color='0.7')
ax.set_xlabel('Simulation $N_\mathrm{zeo}/N_\mathrm{tot}$')
ax.set_ylabel('Predicted $N_\mathrm{zeo}/N_\mathrm{tot}$')
plt.savefig(f"{model_file.rstrip('.pt')}_train.png", format="png", bbox_inches="tight")


fig, ax = plt.subplots(figsize=(4, 3))
for i in range(n_comp):
    ax.scatter(y_test[:, i], y_test_pred[:, i], s=1)
ax.legend(molecules)
ax.plot([0, 1], [0, 1], ls='--', color='0.7')
ax.set_xlabel('Simulation $N_\mathrm{zeo}/N_\mathrm{tot}$')
ax.set_ylabel('Predicted $N_\mathrm{zeo}/N_\mathrm{tot}$')
plt.savefig(f"{model_file.rstrip('.pt')}_test.png", format="png", bbox_inches="tight")