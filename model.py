import numpy as np
import pickle
from sklearn.metrics import f1_score

class ZENA:
    def __init__(self, input_size=30, hidden_size1=128, hidden_size2=32, learning_rate=0.03, iterations=100000, early_stopping_rounds=5000, tolerance=1e-4, init_method='xavier', momentum=0.9):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = 1
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.tolerance = tolerance
        self.init_method = init_method
        self.momentum = momentum
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.init_params()
        self.vW1, self.vb1, self.vW2, self.vb2, self.vW3, self.vb3 = self.init_velocity()

    def init_params(self):
        if self.init_method == 'zero':
            W1 = np.zeros((self.hidden_size1, self.input_size))
            b1 = np.zeros((self.hidden_size1, 1))
            W2 = np.zeros((self.hidden_size2, self.hidden_size1))
            b2 = np.zeros((self.hidden_size2, 1))
            W3 = np.zeros((self.output_size, self.hidden_size2))
            b3 = np.zeros((self.output_size, 1))
        elif self.init_method == 'random':
            W1 = np.random.randn(self.hidden_size1, self.input_size) * 0.01
            b1 = np.random.randn(self.hidden_size1, 1) * 0.01
            W2 = np.random.randn(self.hidden_size2, self.hidden_size1) * 0.01
            b2 = np.random.randn(self.hidden_size2, 1) * 0.01
            W3 = np.random.randn(self.output_size, self.hidden_size2) * 0.01
            b3 = np.random.randn(self.output_size, 1) * 0.01
        elif self.init_method == 'he':
            W1 = np.random.randn(self.hidden_size1, self.input_size) * np.sqrt(2. / self.input_size)
            b1 = np.random.randn(self.hidden_size1, 1) * np.sqrt(2. / self.hidden_size1)
            W2 = np.random.randn(self.hidden_size2, self.hidden_size1) * np.sqrt(2. / self.hidden_size1)
            b2 = np.random.randn(self.hidden_size2, 1) * np.sqrt(2. / self.hidden_size2)
            W3 = np.random.randn(self.output_size, self.hidden_size2) * np.sqrt(2. / self.hidden_size2)
            b3 = np.random.randn(self.output_size, 1) * np.sqrt(2. / self.output_size)
        elif self.init_method == 'xavier':
            W1 = np.random.randn(self.hidden_size1, self.input_size) * np.sqrt(1. / self.input_size)
            b1 = np.random.randn(self.hidden_size1, 1) * np.sqrt(1. / self.hidden_size1)
            W2 = np.random.randn(self.hidden_size2, self.hidden_size1) * np.sqrt(1. / self.hidden_size1)
            b2 = np.random.randn(self.hidden_size2, 1) * np.sqrt(1. / self.hidden_size2)
            W3 = np.random.randn(self.output_size, self.hidden_size2) * np.sqrt(1. / self.hidden_size2)
            b3 = np.random.randn(self.output_size, 1) * np.sqrt(1. / self.output_size)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
        return W1, b1, W2, b2, W3, b3

    def init_velocity(self):
        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = np.zeros_like(self.b2)
        vW3 = np.zeros_like(self.W3)
        vb3 = np.zeros_like(self.b3)
        return vW1, vb1, vW2, vb2, vW3, vb3

    @staticmethod
    def ReLU(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        Y_hat = self.sigmoid(Z3)
        return Z1, A1, Z2, A2, Z3, Y_hat

    @staticmethod
    def ReLU_deriv(Z):
        return Z > 0

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        epsilon = 1e-10
        log_probs = -Y * np.log(Y_hat + epsilon) - (1 - Y) * np.log(1 - Y_hat + epsilon)
        loss = np.sum(log_probs) / m
        return loss
    
    def backward_prop(self, Z1, A1, Z2, A2, Z3, Y_hat, X, Y):
        m = X.shape[1]
            
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
        dZ2 = self.W3.T.dot(dZ3) * self.ReLU_deriv(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2, dW3, db3

    def update_params(self, dW1, db1, dW2, db2, dW3, db3):
        self.vW1 = self.momentum * self.vW1 + (1 - self.momentum) * dW1
        self.vb1 = self.momentum * self.vb1 + (1 - self.momentum) * db1
        self.vW2 = self.momentum * self.vW2 + (1 - self.momentum) * dW2
        self.vb2 = self.momentum * self.vb2 + (1 - self.momentum) * db2
        self.vW3 = self.momentum * self.vW3 + (1 - self.momentum) * dW3
        self.vb3 = self.momentum * self.vb3 + (1 - self.momentum) * db3

        self.W1 -= self.learning_rate * self.vW1
        self.b1 -= self.learning_rate * self.vb1
        self.W2 -= self.learning_rate * self.vW2
        self.b2 -= self.learning_rate * self.vb2
        self.W3 -= self.learning_rate * self.vW3
        self.b3 -= self.learning_rate * self.vb3

    @staticmethod
    def get_predictions(Y_hat):
        return (Y_hat > 0.5).astype(int)
    
    @staticmethod
    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size

    @staticmethod
    def get_f1_score(predictions, Y):
        return f1_score(Y.flatten(), predictions.flatten())
        
    def fit(self, X_train, Y_train, X_val, Y_val):
        train_loss_history = []
        val_loss_history = []
        best_loss = float('inf')
        best_iter = 0

        for i in range(self.iterations):
            Z1, A1, Z2, A2, Z3, Y_hat = self.forward_prop(X_train)
            train_loss = self.compute_loss(Y_hat, Y_train)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(Z1, A1, Z2, A2, Z3, Y_hat, X_train, Y_train)
            self.update_params(dW1, db1, dW2, db2, dW3, db3)

            if i % 1000 == 0:
                print(f"Iteration: {i}")
                train_loss_history.append(train_loss)
                print(f"Training Loss: {train_loss}")

                _, _, _, _, _, val_Y_hat = self.forward_prop(X_val)
                val_loss = self.compute_loss(val_Y_hat, Y_val)
                val_loss_history.append(val_loss)
                print(f"Validation Loss: {val_loss}")

                if val_loss < best_loss - self.tolerance:
                    best_loss = val_loss
                    best_iter = i
                elif i - best_iter >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i}")
                    break

        return train_loss_history, val_loss_history

    def predict(self, X):
        _, _, _, _, _, Y_hat = self.forward_prop(X)
        predictions = self.get_predictions(Y_hat)
        return predictions.flatten()

    def save_params(self, file_path):
        params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3
        }
        with open(file_path, 'wb') as file:
            pickle.dump(params, file)
        print(f"Parameters saved to {file_path}")

    def load_params(self, file_path):
        with open(file_path, 'rb') as file:
            params = pickle.load(file)
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.W3 = params['W3']
        self.b3 = params['b3']
        print(f"Parameters loaded from {file_path}")