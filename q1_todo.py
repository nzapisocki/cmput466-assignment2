from utils import plot_data, generate_data
import numpy as np

"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """

    w = np.zeros(X.shape[1])
    b = 0.0

    learning_rate = 0.1
    epochs = 1000

    for i in range(epochs):
        z = X @ w + b
        y = 1 / (1 + np.exp(-z))
        error = y - t

        grad_w = X.T @ error / len(t)
        grad_b = error.mean()

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    z = X @ w + b

    y = 1 / (1 + np.exp(-z))

    t = (y >= 0.5).astype(int)

    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """

    x_mean = X.mean(axis=0)
    t_mean = t.mean(axis=0)

    X_c = X - x_mean
    t_c = t - t_mean

    w = np.linalg.inv(X_c.T @ X_c) @ (X_c.T @ t_c)

    b = t_mean - x_mean @ w

    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    y = (X @ w) + b

    t = (y >= 0.5).astype(int)

    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """

    tp = list(t).count(1)
    tn = list(t).count(0)

    fp = 0
    fn = 0

    for i in range(len(t)):
        if t_hat[i] and not t[i]:
            fp += 1
        elif not t_hat[i] and t[i]:
            fn += 1

    acc = (tp + tn) / (tp + tn + fp + fn)

    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
