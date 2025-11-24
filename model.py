import numpy as np
import pickle
from sklearn.metrics import f1_score

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class ZENA:
    def __init__(self, input_size=30, hidden_size1=128, hidden_size2=32, learning_rate=0.03, iterations=100000, early_stopping_rounds=5000, tolerance=1e-4, init_method='xavier', momentum=0.9, use_gpu=False):
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
        
        # GPU 설정
        if use_gpu and CUPY_AVAILABLE:
            self.use_gpu = True
            self.xp = cp  # CuPy 사용
            print("GPU acceleration enabled (CuPy)")
        else:
            self.use_gpu = False
            self.xp = np  # NumPy 사용
            if use_gpu and not CUPY_AVAILABLE:
                print("Warning: GPU requested but CuPy not available. Using CPU (NumPy).")
        
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
        
        if self.use_gpu:
            W1 = cp.asarray(W1)
            b1 = cp.asarray(b1)
            W2 = cp.asarray(W2)
            b2 = cp.asarray(b2)
            W3 = cp.asarray(W3)
            b3 = cp.asarray(b3)
        
        return W1, b1, W2, b2, W3, b3

    def init_velocity(self):
        xp = self.xp
        vW1 = xp.zeros_like(self.W1)
        vb1 = xp.zeros_like(self.b1)
        vW2 = xp.zeros_like(self.W2)
        vb2 = xp.zeros_like(self.b2)
        vW3 = xp.zeros_like(self.W3)
        vb3 = xp.zeros_like(self.b3)
        return vW1, vb1, vW2, vb2, vW3, vb3

    def ReLU(self, Z):
        return self.xp.maximum(Z, 0)

    def sigmoid(self, x):
        return 1.0 / (1.0 + self.xp.exp(-x))
    
    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        Y_hat = self.sigmoid(Z3)
        return Z1, A1, Z2, A2, Z3, Y_hat

    def ReLU_deriv(self, Z):
        return Z > 0

    def compute_loss(self, Y_hat, Y):
        xp = self.xp
        # Y_hat shape: (1, n_samples), Y shape: (n_samples, 1) or (1, n_samples)
        # Ensure both are (1, n_samples) for consistent broadcasting
        if len(Y.shape) == 2 and Y.shape[0] > Y.shape[1]:
            # Y is (n_samples, 1), transpose to (1, n_samples)
            Y = Y.T
        elif len(Y.shape) == 1:
            # Y is (n_samples,), reshape to (1, n_samples)
            Y = Y.reshape(1, -1)
        
        m = Y.shape[1]  # number of samples
        epsilon = 1e-10
        log_probs = -Y * xp.log(Y_hat + epsilon) - (1 - Y) * xp.log(1 - Y_hat + epsilon)
        loss = float(xp.sum(log_probs) / m)  # GPU 배열을 Python float로 변환
        return loss
    
    def backward_prop(self, Z1, A1, Z2, A2, Z3, Y_hat, X, Y):
        xp = self.xp
        m = X.shape[1]
        
        # Ensure Y has same shape as Y_hat: (1, n_samples)
        if len(Y.shape) == 2 and Y.shape[0] > Y.shape[1]:
            # Y is (n_samples, 1), transpose to (1, n_samples)
            Y = Y.T
        elif len(Y.shape) == 1:
            # Y is (n_samples,), reshape to (1, n_samples)
            Y = Y.reshape(1, -1)
        
        dZ3 = Y_hat - Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * xp.sum(dZ3, axis=1, keepdims=True)
        dZ2 = self.W3.T.dot(dZ3) * self.ReLU_deriv(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * xp.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * xp.sum(dZ1, axis=1, keepdims=True)
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

    def get_predictions(self, Y_hat):
        if self.use_gpu:
            Y_hat_cpu = cp.asnumpy(Y_hat)
        else:
            Y_hat_cpu = Y_hat
        return (Y_hat_cpu > 0.5).astype(int)
    
    @staticmethod
    def get_accuracy(predictions, Y):
        if CUPY_AVAILABLE and hasattr(Y, 'device'):  # CuPy 배열인 경우
            Y_cpu = cp.asnumpy(Y)
        else:
            Y_cpu = Y
        return np.sum(predictions == Y_cpu) / Y_cpu.size

    @staticmethod
    def get_f1_score(predictions, Y):
        if CUPY_AVAILABLE and hasattr(Y, 'device'):  # CuPy 배열인 경우
            Y_cpu = cp.asnumpy(Y)
        else:
            Y_cpu = Y
        return f1_score(Y_cpu.flatten(), predictions.flatten())
        
    def fit(self, X_train, Y_train, X_val, Y_val):
        if self.use_gpu:
            X_train = cp.asarray(X_train)
            if not isinstance(Y_train, cp.ndarray):
                Y_train = cp.asarray(Y_train)
            if not isinstance(Y_val, cp.ndarray):
                Y_val = cp.asarray(Y_val)
            X_val = cp.asarray(X_val)
        
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
        if self.use_gpu:
            X = cp.asarray(X)
        _, _, _, _, _, Y_hat = self.forward_prop(X)
        predictions = self.get_predictions(Y_hat)
        return predictions.flatten()

    def save_params(self, file_path):
        if self.use_gpu:
            params = {
                'W1': cp.asnumpy(self.W1),
                'b1': cp.asnumpy(self.b1),
                'W2': cp.asnumpy(self.W2),
                'b2': cp.asnumpy(self.b2),
                'W3': cp.asnumpy(self.W3),
                'b3': cp.asnumpy(self.b3)
            }
        else:
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
        
        if self.use_gpu:
            self.W1 = cp.asarray(params['W1'])
            self.b1 = cp.asarray(params['b1'])
            self.W2 = cp.asarray(params['W2'])
            self.b2 = cp.asarray(params['b2'])
            self.W3 = cp.asarray(params['W3'])
            self.b3 = cp.asarray(params['b3'])
        else:
            self.W1 = params['W1']
            self.b1 = params['b1']
            self.W2 = params['W2']
            self.b2 = params['b2']
            self.W3 = params['W3']
            self.b3 = params['b3']
        print(f"Parameters loaded from {file_path}")