import numpy as np
import matplotlib.pyplot as plt

# flow of a simple neural network with one hidden layer
# input -> multiplication with weights -> add bias -> activation function (using ReLU here) -> output layer

np.random.seed(42)
x = np.random.randn(100, 1)
y = 7*x + 11 # the true function 

noise = 0.1 * np.random.randn(100, 1)
y += noise

w1 = np.random.randn(1, 16) # 16 neurons in hidden layer
b1 = np.random.randn(1, 16) 
w2 = np.random.randn(16, 1) # converges to single output
b2 = np.random.randn(1, 1)


def relu(x):
    return np.maximum(0, x)

def forward(x):
    o1 = x @ w1 + b1 # @ is matrix multiplication
    fo1 = relu(o1)
    o2 = fo1 @ w2 + b2
    return o1, fo1, o2

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def relu_derivative(x):
    return (x > 0).astype(float)

def backprop(x, y, o1, fo1, o2):
    global w1, b1, w2, b2

    n = x.shape[0] # number of samples

    dLdy_pred = 2 * (o2 - y) / n

    dw2 = fo1.T @ dLdy_pred
    db2 = np.sum(dLdy_pred, axis=0, keepdims=True)

    dLdfo1 = dLdy_pred @ w2.T
    dLdo1 = dLdfo1 * relu_derivative(o1)

    dw1 = x.T @ dLdo1
    db1 = np.sum(dLdo1, axis=0, keepdims=True)

    return dw1, db1, dw2, db2   

lr =float(input("Enter the learning rate: "))
epochs = int(input("Enter the number of epochs: "))

for epoch in range(epochs):

    o1, fo1, o2 = forward(x)
    loss = mse(o2, y)

    dw1, db1, dw2, db2 = backprop(x, y, o1, fo1, o2)

    w1 -= lr* dw1
    b1 -= lr* db1
    w2 -= lr* dw2
    b2 -= lr* db2

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    epoch +=1

_, _, y_pred = forward(x)
plt.scatter(x, y, label="True Data")
plt.scatter(x, y_pred, color='r', label="Predicted Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Predictions")
plt.legend()
plt.show()

