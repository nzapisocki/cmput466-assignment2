# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows * ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows * ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]), test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here

    # ------ prediction ------ #
    z = X @ W

    y = []

    for i in range(len(z)):
        exp_scores = np.exp(z[i] - np.max(z[i]))
        answer = exp_scores / exp_scores.sum()
        y.append(answer)

    # --- t_hat --- #

    t_hat = np.argmax(y, axis=1)

    # --- loss --- #
    loss = 0

    for m in range(len(y)):
        true_class = t[m, 0]
        loss -= np.log(y[m][true_class])

    # -- accuracy -- #

    acc = np.mean(t_hat == t[:, 0])

    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # TODO Your code here

    train_losses = []
    valid_accs = []
    acc_best = -1

    W = np.zeros((X_train.shape[1], N_class))

    for epoch in range(1, MaxEpoch + 1):
        order = np.random.permutation(N_train)

        for start in range(0, N_train, batch_size):
            indices = order[start:start + batch_size]
            X_batch = X_train[indices]
            t_batch = y_train[indices]

            y, _, _, _ = predict(X_batch, W, t_batch)
            error = np.array(y)

            for m in range(len(t_batch)):
                true_class = t_batch[m, 0]
                error[m, true_class] -= 1

            gradient = X_batch.T @ error / len(X_batch)
            W -= alpha * gradient

        _, _, train_loss, _ = predict(X_train, W, y_train)
        _, _, _, val_acc = predict(X_val, W, t_val)
        train_losses.append(train_loss / N_train)
        valid_accs.append(val_acc)

        if val_acc > acc_best:
            acc_best = val_acc
            epoch_best = epoch
            W_best = W.copy()

    return epoch_best, acc_best, W_best, train_losses, valid_accs


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()

print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)

N_class = 10

alpha = 0.1  # learning rate
batch_size = 100  # batch size
MaxEpoch = 50  # Maximum epoch
decay = 0.  # weight decay

# TODO: report 3 number, plot 2 curves
epoch_best, acc_best, W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)

print("Best epoch:", epoch_best)
print("Validation accuracy:", acc_best)
print("Test accuracy:", acc_test)

epochs = range(1, MaxEpoch + 1)

plt.figure()
plt.plot(epochs, train_losses)
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.savefig("training_loss.png")

plt.figure()
plt.plot(epochs, valid_accs)
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.savefig("validation_accuracy.png")

plt.show()

