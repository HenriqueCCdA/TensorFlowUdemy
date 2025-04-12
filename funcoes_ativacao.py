import numpy as np

def step_function(soma):
    return int(soma >= 1)

def sigmoid_function(soma):
    return 1/ (1 + np.exp(-soma))

def tahn_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def relu_function(soma):
    return max(soma, 0.0)

def linear_function(soma):
    return soma

def soft_max_function(x):
    ex = np.exp(x)
    return ex / ex.sum()

valores = (
    step_function(-0.358),
    sigmoid_function(-0.358),
    tahn_function(-0.358),
    relu_function(-0.358),
    linear_function(-0.358),
    soft_max_function([5.0, 2.0, 1.3]),
)
print(valores)


def percepton(w, x):
    return np.dot(w, x)

w = np.array([0.2, 0.5, 0.1])
x = np.array([5.0, 2.0, 1.0])


print(sigmoid_function(percepton(w, x)))
print(tahn_function(percepton(w, x)))
print(relu_function(percepton(w, x)))
print(linear_function(percepton(w, x)))

classe = np.array([1, 0, 1, 0])
previsto = np.array([0.3, 0.02, 0.89, 0.32])

def mean_absolute_error(y, x):
    return np.abs(y-x).sum() / len(y)

def mean_square_error(y, x):
    return np.power(y-x, 2).sum() / len(y)

def root_mean_square_error(y, x):
    return np.square(np.power(y-x, 2).sum() / len(y))

print(mean_absolute_error(classe, previsto))
print(mean_square_error(classe, previsto))
print(root_mean_square_error(classe, previsto))
