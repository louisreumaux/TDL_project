import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nn_utils import training, test, train

def nn_training_sizes(n_samples_list, input_training, z_training, input_test, z_test):
    test_error_nn = []
    X_test = torch.tensor(input_test, dtype=torch.float32)
    z_test = torch.tensor(z_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test, z_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    for i in range(len(n_samples_list)):
        n_samples = n_samples_list[i]
        z_train = z_training[:n_samples]
        input_train = input_training[:n_samples]

        X_train = torch.tensor(input_train, dtype=torch.float32)
        Y_train = torch.tensor(z_train, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        learning_rate = 0.005
        min_test_error = training(train_loader, test_loader, 512, learning_rate)
        test_error_nn.append(min_test_error)
    
    return test_error_nn