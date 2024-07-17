import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from dataset import get_train_test_splits, make_datasets

def langmuir(
    x_fit,
    logq0, logq1, logq2, logq3, logq4, logq5, logq6, logq7, 
    s0, s1, s2, s3, s4, s5, s6, s7,
    h0, h1, h2, h3, h4, h5, h6, h7,
):
    invT = x_fit[8]
    p0 = x_fit[0]
    e0 = np.exp(h0 * invT + s0) * p0
    p1 = x_fit[1]
    e1 = np.exp(h1 * invT + s1) * p1
    p2 = x_fit[2]
    e2 = np.exp(h2 * invT + s2) * p2
    p3 = x_fit[3]
    e3 = np.exp(h3 * invT + s3) * p3
    p4 = x_fit[4]
    e4 = np.exp(h4 * invT + s4) * p4
    p5 = x_fit[5]
    e5 = np.exp(h5 * invT + s5) * p5
    p6 = x_fit[6]
    e6 = np.exp(h6 * invT + s6) * p6
    p7 = x_fit[7]
    e7 = np.exp(h7 * invT + s7) * p7

    e_tot = (1 + e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7)

    n0 = np.exp(logq0) * e0 / e_tot
    n1 = np.exp(logq1) * e1 / e_tot
    n2 = np.exp(logq2) * e2 / e_tot
    n3 = np.exp(logq3) * e3 / e_tot
    n4 = np.exp(logq4) * e4 / e_tot
    n5 = np.exp(logq5) * e5 / e_tot
    n6 = np.exp(logq6) * e6 / e_tot
    n7 = np.exp(logq7) * e7 / e_tot
    return np.concatenate([n0, n1, n2, n3, n4, n5, n6, n7], 0)



paths = ["data-full/MFI-0.csv", "data-binary-ternary/MFI-0.csv"]


df_train, df_test, n_comp = get_train_test_splits(
    paths,
    (1, 2, 3),
    (6, 7, 8)
)
data_train, data_test = make_datasets(df_train, df_test, n_comp)
x_train, y_train, x_test, y_test = (
    t.numpy() for t in data_train.tensors + data_test.tensors
)

x_train_flat = np.concatenate([
    np.where(x_train[:, :, 1] == -1, np.zeros_like(x_train[:, :, 0]), np.exp(x_train[:, :, 0])),
    np.max(x_train[:, :, 1], axis=1, keepdims=True)
], 1)
x_test_flat = np.concatenate([
    np.where(x_test[:, :, 1] == -1, np.zeros_like(x_test[:, :, 0]), np.exp(x_test[:, :, 0])),
    np.max(x_test[:, :, 1], axis=1, keepdims=True)
], 1)
print(x_train_flat.shape, y_train.shape)
popt, pcov = curve_fit(langmuir, x_train_flat.T, y_train.T.ravel())

y_train_pred = np.array(langmuir(x_train_flat.T, *popt)).reshape(8, -1).T

y_test_pred = np.array(langmuir(x_test_flat.T, *popt)).reshape(8, -1).T


molecules = ['pX', 'oX', 'mX', 'bz', 'tol', 'eb', 'H2', 'eth']
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
plt.savefig("langmuir_train.png", format="png", bbox_inches="tight")


fig, ax = plt.subplots(figsize=(4, 3))
for i in range(n_comp):
    ax.scatter(y_test[:, i], y_test_pred[:, i], s=1)
ax.legend(molecules)
ax.plot([0, 1], [0, 1], ls='--', color='0.7')
ax.set_xlabel('Simulation $N_\mathrm{zeo}/N_\mathrm{tot}$')
ax.set_ylabel('Predicted $N_\mathrm{zeo}/N_\mathrm{tot}$')
plt.savefig("langmuir_test.png", format="png", bbox_inches="tight")
