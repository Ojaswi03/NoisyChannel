


import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, hinge_loss
from data_loader import mnistLoader

# ----- Loss Functions -----
def hinge_loss(outputs, labels): # labels should be -1 or +1
    return torch.mean(torch.clamp(1 - outputs.squeeze() * labels, min=0))


"""
Summary: compute_gradients computes the gradient of the hinge loss with respect to the weights.
Inputs:
    - inputs: The input data (features).
    - outputs: The model outputs (predictions).
    - labels: The true labels.
Outputs:
    - grad: The computed gradients.
Description:
    - The function calculates the margin between the outputs and labels.
    - It creates a mask to identify which samples contribute to the loss.
    - It computes the gradients based on the masked outputs and inputs.
    - Finally, it returns the computed gradients.
    - The gradients are averaged over the batch size."""
def compute_gradients(inputs, outputs, labels): 
    margin = 1 - outputs * labels
    grad_mask = (margin > 0).float()
    grad_outputs = -labels * grad_mask
    grad = torch.matmul(inputs.T, grad_outputs) / inputs.size(0)
    return grad


# ----- Loss Computation -----
"""
Summary: compute_loss computes the loss and gradients based on the specified loss type.
Inputs:
    - inputs: The input data (features).
    - weights: The model weights.
    - targets: The true labels.
    - loss_type: The type of loss function to use ("hinge" or "cosine").
Outputs:
    - loss: The computed loss.
    - grad: The computed gradients.
Description:
    - If loss_type is "hinge", it computes the hinge loss and gradients.
    - If loss_type is "cosine", it computes the cosine similarity and loss.
    - It returns the computed loss and gradients.
    - The gradients are averaged over the batch size.
"""

def compute_loss(inputs, weights, targets, loss_type="hinge"):
    if loss_type == "hinge":
        outputs = torch.matmul(inputs, weights)
        loss = hinge_loss(outputs.squeeze(), targets.squeeze())
        grad = compute_gradients(inputs, outputs, targets)
        return loss, grad
    elif loss_type == "cosine":
        inputs_norm = torch.nn.functional.normalize(inputs, dim=1)
        weights_norm = torch.nn.functional.normalize(weights, dim=0)
        cos_sim = torch.matmul(inputs_norm, weights_norm).squeeze()
        targets = targets.view(-1)
        loss = torch.mean(1 - targets * cos_sim)
        grad = -torch.matmul(inputs_norm.T, targets.view(-1, 1)) / inputs.size(0)
        return loss, grad
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ----- Evaluation -----

"""
Summary: evaluate_model evaluates the model's accuracy on the test set.
Inputs:
    - weights: The model weights.
    - test_loader: The DataLoader for the test set.
Outputs:
    - accuracy: The computed accuracy of the model on the test set.
Description:
    - The function iterates through the test set in batches.
    - It computes the model's predictions and compares them to the true labels.
    - It calculates the total number of correct predictions and divides by the total number of samples.
    - Finally, it returns the computed accuracy.
"""
def evaluate_model(weights, test_loader):
    correct, total = 0, 0
    for inputs, targets in test_loader:
        inputs = inputs.view(inputs.size(0), -1) # Flatten the input
        logits = torch.matmul(inputs, weights).squeeze() # Compute logits by multiplying inputs with weights
        preds = (logits > 0).long() # Convert logits to binary predictions
        targets = ((targets + 1) // 2).long() # Convert [-1, 1] to [0, 1]
        correct += (preds == targets).sum().item() # Count correct predictions
        total += targets.size(0) 
    return correct / total 

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# # ----- Centralized Training -----
# """
# Summary: centralized_ebm trains a centralized model using the EBM approach.
# Inputs:
#     - train_loader: The DataLoader for the training set.
#     - test_loader: The DataLoader for the test set.
#     - num_features: The number of features in the input data.
#     - sigma: The noise level for robust training.
#     - lr: The learning rate for optimization.
#     - epochs: The number of training epochs.
#     - lambd: The regularization parameter (default is 0.001).
#     - loss_type: The type of loss function to use ("hinge" or "cosine").
# Outputs:
#     - acc_curve: A list of accuracies for each epoch.
#     - loss_curve: A list of losses for each epoch.
# Description:
#     - The function initializes weights and lists to store accuracies and losses.
#     - It iterates through the training set for the specified number of epochs.
#     - For each batch, it computes the loss and gradients.
#     - It applies robust regularization and weight decay.
#     - It updates the weights using the computed gradients.
#     - It evaluates the model on the test set and stores the accuracy and loss.
#     - Finally, it returns the accuracy and loss curves.
#     - For centralzi
    
#  """   
    
# def centralized_ebm(train_loader, test_loader, num_features, sigma, lr, epochs, lambd=0.001, loss_type="hinge"):
#     weights = torch.zeros((num_features, 1), dtype=torch.float32)
#     acc_curve, loss_curve = [], []

#     for epoch in range(epochs):
#         total_loss = 0
#         for inputs, targets in train_loader:
#             inputs = inputs.view(inputs.size(0), -1)
#             targets = targets.view(-1, 1).float()

#             loss, grad = compute_loss(inputs, weights, targets, loss_type)
#             reg = sigma ** 2 * torch.sum(grad ** 2)
#             decay = lambd * torch.norm(weights) ** 2

#             robust_loss = loss + reg + decay
#             robust_grad = grad + 2 * sigma ** 2 * grad + 2 * lambd * weights

#             weights = weights - lr * robust_grad
#             total_loss += robust_loss.item()

#         avg_loss = total_loss / len(train_loader)
#         acc = evaluate_model(weights, test_loader)
#         acc_curve.append(acc)
#         loss_curve.append(avg_loss)
#         print(f"[Centralized EBM] Epoch {epoch+1}: Accuracy={acc:.4f}, Loss={avg_loss:.4f}")

#     return acc_curve, loss_curve

# ----- Centralized SVM Training -----
"""
Summary: centralized_svm trains a centralized baseline using scikit-learn's LinearSVC.
Inputs:
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for testing data.
    - num_features: Number of features in the input.
    - sigma: Not used, placeholder for consistency.
    - lr: Not used, placeholder for consistency.
    - epochs: Not used, placeholder for consistency.
    - lambd: Used to set regularization strength C = 1 / lambd.
    - loss_type: Not used here, sklearn uses hinge loss.
Outputs:
    - acc: Accuracy on the test set.
    - loss: Average hinge loss on training set.
Description:
    - This function collects all training and testing data from the loaders.
    - It trains a LinearSVC on the full dataset with hinge loss.
    - Evaluates final test accuracy.
    - Computes hinge loss on training data as the "loss" metric.
"""
def centralized_svm(train_loader, test_loader, num_features, sigma, lr, epochs, lambd=0.01, loss_type="hinge"):

    X_train, y_train = [], []
    X_test, y_test = [], []

    for inputs, targets in train_loader:
        inputs = inputs.view(inputs.size(0), -1).numpy()
        targets = ((targets.numpy() + 1) // 2).astype(int)
        X_train.append(inputs)
        y_train.append(targets)

    for inputs, targets in test_loader:
        inputs = inputs.view(inputs.size(0), -1).numpy()
        targets = ((targets.numpy() + 1) // 2).astype(int)
        X_test.append(inputs)
        y_test.append(targets)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    # Set regularization strength: C = 1 / lambd
    # clf = LinearSVC(C=1.0/lambd, loss='hinge', max_iter=10000, dual=False)
    clf = LinearSVC(C=1.0/lambd, loss='hinge', max_iter=10000, dual=True, verbose=1)

    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    train_preds = clf.decision_function(X_train)
    avg_loss = hinge_loss(y_train, train_preds)

    return acc, avg_loss

# ----- Federated Training -----
def train_federated(client_loaders, test_loader, num_features, sigma, lr, epochs, robust=False, num_inner_updates=1, loss_type="hinge", lambd=0.01):
    global_weights = torch.zeros((num_features, 1))
    acc_curve, loss_curve = [], []

    # Initialize persistent G_prev for each client
    G_prev_list = [torch.zeros_like(global_weights) for _ in range(len(client_loaders))]

    for epoch in range(epochs):
        rho_t = 1 / (epoch + 1) ** 0.7
        # gamma_t = 1 / (epoch + 1) ** 0.9
        gamma_t = 0.5   # 2nd option
        local_models = []
        total_loss = 0

        for client_idx, loader in enumerate(client_loaders):
            w_local = global_weights.clone()
            w_noisy = w_local + torch.normal(mean=0, std=sigma, size=w_local.shape) if robust else w_local.clone()
            w_prev = w_noisy.clone()

            G_prev = G_prev_list[client_idx]  # Load this client's memory

            for _ in range(num_inner_updates):
                for inputs, targets in loader:
                    inputs = inputs.view(inputs.size(0), -1)
                    targets = targets.view(-1, 1).float()

                    loss, grad = compute_loss(inputs, w_noisy, targets, loss_type)

                    if robust:
                        # EBM update with noise
                        G_t = (1 - rho_t) * G_prev + rho_t * grad
                        surrogate_grad = 2 * lambd * (w_noisy - w_prev) + G_t
                        w_star = w_noisy - lr * surrogate_grad
                        w_noisy = w_noisy + gamma_t * (w_star - w_noisy)
                        # w_noisy  = w_noisy - lr + surrogate_grad #1st option

                        G_prev = G_t  # update local memory
                    else:
                        # FedAvg update
                        w_noisy = w_noisy - lr * grad

                    total_loss += loss.item()

            G_prev_list[client_idx] = G_prev.clone()  # update local memory of this client
            local_models.append(w_noisy)

        global_weights = torch.stack(local_models).mean(dim=0)
        acc = evaluate_model(global_weights, test_loader)
        acc_curve.append(acc)
        loss_curve.append(total_loss / len(client_loaders))

        print(f"[{'Federated EBM' if robust else 'FedAvg'}] Epoch {epoch+1}: Accuracy = {acc:.4f}, Loss = {loss_curve[-1]:.4f}")

    return acc_curve, loss_curve


# ----- Plotting -----
def plot_results(acc_ebm, loss_ebm, acc_fl, loss_fl, acc_cent, loss_cent, folder):
    os.makedirs(folder, exist_ok=True)
    acc_ebm = moving_average(acc_ebm)
    acc_fl = moving_average(acc_fl)
    loss_ebm = moving_average(loss_ebm)
    loss_fl = moving_average(loss_fl)
    epochs = range(1, len(acc_ebm) + 1)

    plt.figure()
    plt.plot(epochs, acc_cent[:len(epochs)], label="Centralized EBM", marker='^')
    plt.plot(epochs, acc_ebm, label="Proposed Federated Learning (EBM)", marker='o')
    plt.plot(epochs, acc_fl, label="Conventional Federated Learning (FedAvg)", marker='s')
    plt.title("Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "accuracy_comparison.png"), dpi=300)

    plt.figure()
    plt.semilogy(epochs, loss_fl, label="Conventional Federated Learning (FedAvg)", marker='s')
    plt.semilogy(epochs, loss_ebm, label="Proposed Federated Learning (EBM)", marker='o')
    plt.semilogy(epochs, loss_cent[:len(epochs)], label="Centralized SVM (sklearn)", marker='^')
    plt.title("Loss Comparison (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, which='both', axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "loss_comparison.png"), dpi=300)
    plt.close('all')

# ----- Main -----
def main():
    start_time = time.time()
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print("Starting EBM training...")
    try:
        dataset_choice = input("Select dataset (MNIST only for now): ").strip().lower()
        epochs = int(input("Enter number of epochs (default=32): ") or 32)
        batch_size = int(input("Enter batch size (default=64): ") or 64)
        sigma = float(input("Enter communication noise level sigma (default=0.0005): ") or 0.0005)
        lr = float(input("Enter learning rate (default=0.05): ") or 0.05)
        lambd = float(input("Enter lambda for weight regularization (default=0.01): ") or 0.01)
        num_clients = int(input("Enter number of clients (default=5): ") or 5)
        num_inner_updates = int(input("Enter number of inner updates per client (default=1): ") or 1)
        loss_type = input("Select loss type (hinge/cosine) [default=hinge]: ").strip().lower() or "hinge"
    except ValueError:
        print("Invalid input. Using default settings.")
        dataset_choice, num_clients, batch_size, epochs, lr, sigma, num_inner_updates, loss_type = "mnist", 5, 64, 32, 0.0005, 0.05, 1, "hinge"

    if dataset_choice != "mnist":
        raise NotImplementedError("Only MNIST supported for this script.")
    
    """
    FIXED: binary=True ensures ±1 labels for hinge loss
        
    """

    train_loader, test_loader = mnistLoader(batch_size=batch_size, binary=True)
    num_features = 28 * 28
    dataset = list(train_loader.dataset)
    split_size = len(dataset) // num_clients
    # Split the dataset into num_clients parts
    # and create DataLoader for each client
    # FIXED: Use DataLoader to create client datasets
    client_loaders = [
        torch.utils.data.DataLoader(dataset[i * split_size:(i + 1) * split_size], batch_size=batch_size, shuffle=True)
        for i in range(num_clients)
    ]
    acc_cent, loss_cent = centralized_svm(train_loader, test_loader, num_features, sigma, lr, epochs, lambd=lambd, loss_type=loss_type)
    acc_ebm, loss_ebm = train_federated(client_loaders, test_loader, num_features, sigma, lr, epochs=epochs, robust=True, num_inner_updates=num_inner_updates, loss_type=loss_type, lambd=lambd)
    acc_fl, loss_fl = train_federated(client_loaders, test_loader, num_features, sigma, lr, epochs=epochs, robust=False, num_inner_updates=num_inner_updates, loss_type=loss_type, lambd=lambd)
    plot_results(acc_ebm, loss_ebm, acc_fl, loss_fl, acc_cent, loss_cent, folder="diagram-EBM")
    
    total_time = time.time()- start_time
    if (total_time > 60):
        print(f"Training completed in {total_time/60:.2f} minutes.")
    elif (total_time > 3600):
        print(f"Training completed in {total_time/3600:.2f} hours.")
    else:   
        print(f"Training completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()