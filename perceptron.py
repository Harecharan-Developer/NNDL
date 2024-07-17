import random

def activation(z):
    return 1 if z >= 0 else 0

def perceptron_train(X, y, epochs=10, learning_rate=0.01):
    w = [random.uniform(-0.01, 0.01) for _ in range(len(X[0]))]
    b = random.uniform(-0.01, 0.01)
    
    def has_converged(X, y, w, b):
        return sum(abs(y[i] - activation(sum(w[j] * X[i][j] for j in range(len(X[0]))))) for i in range(len(X))) < 0.001
    
    for epoch in range(epochs):
        for i in range(len(X)):
            z = sum(w[j] * X[i][j] for j in range(len(X[0]))) + b
            y_hat = activation(z)
            e = y[i] - y_hat

            for j in range(len(w)):
                w[j] += learning_rate * e * X[i][j]
            b += learning_rate * e
        
        if has_converged(X, y, w, b):
            break
    
    return w, b

def perceptron_predict(X, w, b):
    predictions = []
    for i in range(len(X)):
        z = sum(w[j] * X[i][j] for j in range(len(X[0]))) + b
        y_hat = activation(z)
        predictions.append(y_hat)
    return predictions

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]  
w, b = perceptron_train(X, y, epochs=10, learning_rate=0.1)
print("Weights:", w)
print("Bias:", b)

predictions = perceptron_predict(X, w, b)
print("Predictions:", predictions)
