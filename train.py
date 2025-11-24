import argparse
from sklearn.metrics import f1_score, accuracy_score
from model import ZENA
import pandas as pd
import numpy as np

def data_splitter(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):

    if train_ratio+val_ratio+test_ratio != 1 :
        print(f"Ratio setting is not correct : {train_ratio}+{val_ratio}+{test_ratio} != 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    print(f"Total data count : {len(data)}")
    
    # Mixing up indices
    shuffled_indices = np.random.permutation(data.index)

    # Set the split point
    train_split_point = int(len(data)*train_ratio)
    validation_split_point = int(len(data)*(train_ratio+val_ratio))

    # Separate indices
    train_indices = shuffled_indices[:train_split_point]
    val_indices = shuffled_indices[train_split_point:validation_split_point]
    test_indices = shuffled_indices[validation_split_point:]

    # Assign df data according to the indices
    train_data = data.loc[train_indices].reset_index(drop=True)
    val_data = data.loc[val_indices].reset_index(drop=True)
    test_data = data.loc[test_indices].reset_index(drop=True)

    print(train_data['Labels'].value_counts())
    print(val_data['Labels'].value_counts())
    print(test_data['Labels'].value_counts())
    print('*'*100)
    
    # Split X, Y
    train_X = np.array(train_data.iloc[:, :-1], dtype=int).T
    train_Y = np.array(train_data.iloc[:, -1], dtype=int)
    val_X = np.array(val_data.iloc[:, :-1], dtype=int).T
    val_Y = np.array(val_data.iloc[:, -1], dtype=int)
    test_X = np.array(test_data.iloc[:, :-1], dtype=int).T
    test_Y = np.array(test_data.iloc[:, -1], dtype=int)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def get_confusion_matrix(model, test_x, test_y):
    yhat_test = model.predict(test_x)
    accuracy = accuracy_score(test_y, yhat_test)
    f1 = f1_score(test_y, yhat_test, average='weighted')
    return accuracy, f1

parser = argparse.ArgumentParser(prog="ZENA model trainer", description="Train a double-layer binary classification model")
g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_name", type=str, required=False, default="ZENA", help="Specify the model name")
g.add_argument("--input_size", type=int, required=False, default=30, help="Number of features")
g.add_argument("--hidden_size1", type=int, required=False, default=128, help="Determines the number of parameters for the first hidden layer")
g.add_argument("--hidden_size2", type=int, required=False, default=32, help="Determines the number of parameters for the second hidden layer")
g.add_argument("--learning_rate", type=float, required=False, default=0.03, help="Default 0.03. Set the learning rate.")
g.add_argument("--iterations", type=int, required=False, default=100000, help="Default set at 100000")
g.add_argument("--early_stopping_rounds", type=int, default=5000, required=False, help="Default set at 5000")
g.add_argument("--tolerance", type=float, required=False, default=1e-4, help="Default set at 1e-4")
g.add_argument("--init_method", type=str, required=False, default='xavier', help="Default xavier, choose one from zero, random, he, and xavier")
g.add_argument("--momentum", type=float, required=False, default=0.9, help="Default set at 0.9")
g.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration (requires CuPy)")
g.add_argument("--note", type=str, required=False, default="", help="Any note to mention for this checkpoint")


def main(args):
    model_name = args.model_name
    input_size = args.input_size
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    learning_rate = args.learning_rate
    iterations = args.iterations
    early_stopping_rounds = args.early_stopping_rounds
    tolerance = args.tolerance
    init_method = args.init_method
    momentum = args.momentum
    use_gpu = args.use_gpu
    note = args.note
    
    # GPU support check
    if use_gpu:
        try:
            import cupy as cp
            print("GPU acceleration enabled (CuPy available)")
        except ImportError:
            print("Warning: --use_gpu specified but CuPy is not installed. Falling back to CPU.")
            print("To use GPU acceleration, install CuPy: pip install cupy-cuda11x (for CUDA 11.x)")
            use_gpu = False

    # Data Loader
    data = pd.read_csv("ZENA_data.csv")
    train_x, train_y, val_x, val_y, test_x, test_y = data_splitter(data)

    # Model Init
    model = ZENA(input_size=input_size,
                    hidden_size1=hidden_size1,
                    hidden_size2=hidden_size2,
                    learning_rate=learning_rate,
                    iterations=iterations,
                    early_stopping_rounds=early_stopping_rounds,
                    tolerance=tolerance,
                    init_method=init_method,
                    momentum=momentum,
                    use_gpu=use_gpu)

    # Model Train
    train_loss_history, val_loss_history = model.fit(train_x, train_y, val_x, val_y)
    accuracy, f1 = get_confusion_matrix(model, test_x, test_y)
    
    # Get Experiment Results
    experiment_record = {
        'model_name': model_name,
        'input_size': input_size,
        'learning_rate': learning_rate,
        'iterations': iterations,
        'actual_iterations': len(train_loss_history) * 1000,
        'min_val_loss': min(val_loss_history),
        'final_train_loss': train_loss_history[-1],
        'final_val_loss': val_loss_history[-1],
        'test_accuracy': accuracy,
        'test_f1': f1,
        'note': note
    }

    # Save the Results
    try:
        experiment_result = pd.read_csv('model_experiment.csv')
        experiment_result = pd.concat([experiment_result, pd.DataFrame([experiment_record])], ignore_index=True)
    except FileNotFoundError:
        experiment_result = pd.DataFrame([experiment_record])
        
    experiment_result.to_csv('model_experiment.csv', index=False)
    save_path = f'params/{model_name}.pkl'
    model.save_params(save_path)
    print(f"All process completed: model saved at {save_path}")

if __name__ == "__main__":
    exit(main(parser.parse_args()))