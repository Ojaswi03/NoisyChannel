
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from data_loader import mnistLoader, cifar10Loader

# # ----- Core functions -----
# def softmax(logits):
#     exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
#     return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

# def cross_entropy_loss(predictions, targets):
#     batch_size = predictions.shape[0]
#     log_preds = torch.log(predictions + 1e-9)
#     return -torch.sum(targets * log_preds) / batch_size

# def compute_gradients(inputs, predictions, targets):
#     batch_size = inputs.shape[0]
#     grad_logits = predictions - targets
#     return torch.matmul(inputs.T, grad_logits) / batch_size

# def evaluate_model(weights, test_loader, num_classes):
#     correct, total = 0, 0
#     for inputs, targets in test_loader:
#         inputs = inputs.view(inputs.size(0), -1)
#         logits = torch.matmul(inputs, weights)
#         preds = softmax(logits)
#         correct += (torch.argmax(preds, dim=1) == targets).sum().item()
#         total += targets.size(0)
#     return correct / total

# def moving_average(data, window_size=3):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# # ----- Centralized Training -----
# def train_centralized(train_loader, test_loader, num_features, num_classes, epochs, lr):
#     weights = torch.zeros((num_features, num_classes))
#     acc_curve, loss_curve = [], []

#     for epoch in range(epochs):
#         total_loss = 0
#         for inputs, targets in train_loader:
#             inputs = inputs.view(inputs.size(0), -1)
#             targets_one_hot = torch.eye(num_classes)[targets]

#             logits = torch.matmul(inputs, weights)
#             preds = softmax(logits)
#             loss = cross_entropy_loss(preds, targets_one_hot)
#             grad = compute_gradients(inputs, preds, targets_one_hot)

#             weights -= lr * grad
#             total_loss += loss.item()

#         acc = evaluate_model(weights, test_loader, num_classes)
#         acc_curve.append(acc)
#         loss_curve.append(total_loss / len(train_loader))
#         print(f"[Centralized] Epoch {epoch+1}: Accuracy = {acc:.4f}, Loss = {loss_curve[-1]:.4f}")

#     return acc_curve, loss_curve

# # ----- Federated Training -----
# def train_federated(client_loaders, test_loader, num_features, num_classes, sigma, lr, epochs, robust=False):
#     global_weights = torch.zeros((num_features, num_classes))
#     acc_curve, loss_curve = [], []

#     for epoch in range(epochs):
#         local_weights = []
#         total_loss = 0

#         for loader in client_loaders:
#             w = global_weights.clone()
#             for inputs, targets in loader:
#                 inputs = inputs.view(inputs.size(0), -1)
#                 targets_one_hot = torch.eye(num_classes)[targets]

#                 logits = torch.matmul(inputs, w)
#                 preds = softmax(logits)
#                 loss = cross_entropy_loss(preds, targets_one_hot)
#                 grad = compute_gradients(inputs, preds, targets_one_hot)

#                 if robust:
#                     grad_norm_sq = torch.sum(grad ** 2)
#                     loss += sigma ** 2 * grad_norm_sq
#                     grad += 2 * sigma ** 2 * grad

#                 w = w - lr * grad
#                 total_loss += loss.item()

#             local_weights.append(w)

#         global_weights = torch.stack(local_weights).mean(dim=0)
#         acc = evaluate_model(global_weights, test_loader, num_classes)
#         acc_curve.append(acc)
#         loss_curve.append(total_loss / len(client_loaders))

#         print(f"[{'Federated EBM' if robust else 'FedAvg'}] Epoch {epoch+1}: Accuracy = {acc:.4f}, Loss = {loss_curve[-1]:.4f}")

#     return acc_curve, loss_curve

# # ----- Plotting -----
# def plot_results(acc_ebm, loss_ebm, acc_fl, loss_fl, acc_cent, loss_cent, folder):
#     os.makedirs(folder, exist_ok=True)

#     # Smooth the curves slightly
#     acc_ebm = moving_average(acc_ebm)
#     acc_fl = moving_average(acc_fl)
#     acc_cent = moving_average(acc_cent)

#     loss_ebm = moving_average(loss_ebm)
#     loss_fl = moving_average(loss_fl)
#     loss_cent = moving_average(loss_cent)

#     epochs = range(1, len(acc_ebm) + 1)

#     # Plot Accuracy
#     plt.figure()
#     plt.plot(epochs, acc_cent, label="Centralized Learning", marker='^')
#     plt.plot(epochs, acc_ebm, label="Proposed Federated Learning (EBM)", marker='o')
#     plt.plot(epochs, acc_fl, label="Conventional Federated Learning (FedAvg)", marker='s')
#     plt.title("Accuracy Comparison")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder, "accuracy_comparison.png"), dpi=300)

#     # Plot Loss (log scale)
#     plt.figure()
#     plt.semilogy(epochs, loss_fl, label="Conventional Federated Learning (FedAvg)", marker='s')
#     plt.semilogy(epochs, loss_ebm, label="Proposed Federated Learning (EBM)", marker='o')
#     plt.semilogy(epochs, loss_cent, label="Centralized Learning", marker='^')
#     plt.title("Loss Comparison (Log Scale)")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss (log scale)")
#     plt.grid(True, which='both', axis='y')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder, "loss_comparison.png"), dpi=300)

#     plt.close('all')

# # ----- Main -----
# def main():
#     try:
#         dataset_choice = input("Select dataset (MNIST/CIFAR10): ").strip().lower()
#         epochs = int(input("Enter number of epochs (default=10): ") or 10)
#         batch_size = int(input("Enter batch size (default=64): ") or 64)
#         sigma = float(input("Enter communication noise level sigma (default=0.1): ") or 0.1)
#         lr = float(input("Enter learning rate (default=0.1): ") or 0.1)
#         num_clients = int(input("Enter number of clients (default=5): ") or 5)
#     except ValueError:
#         print("Invalid input. Using default settings.")
#         dataset_choice, num_clients, batch_size, epochs, lr, sigma = "mnist", 5, 64, 10, 0.1, 0.1

#     if dataset_choice == "mnist":
#         train_loader, test_loader = mnistLoader(batch_size=batch_size)
#         num_features = 28 * 28
#         num_classes = 10
#     elif dataset_choice == "cifar10":
#         train_loader, test_loader = cifar10Loader(batch_size=batch_size)
#         num_features = 3 * 32 * 32
#         num_classes = 10
#     else:
#         print("Invalid dataset choice. Defaulting to MNIST.")
#         train_loader, test_loader = mnistLoader(batch_size=batch_size)
#         num_features = 28 * 28
#         num_classes = 10

#     dataset = list(train_loader.dataset)
#     split_size = len(dataset) // num_clients
#     client_loaders = [
#         torch.utils.data.DataLoader(dataset[i * split_size:(i + 1) * split_size], batch_size=batch_size, shuffle=True)
#         for i in range(num_clients)
#     ]

#     acc_ebm, loss_ebm = train_federated(client_loaders, test_loader,
#                                         num_features, num_classes, sigma, lr,
#                                         epochs=epochs, robust=True)

#     acc_fl, loss_fl = train_federated(client_loaders, test_loader,
#                                       num_features, num_classes, sigma, lr,
#                                       epochs=epochs, robust=False)

#     acc_cent, loss_cent = train_centralized(train_loader, test_loader,
#                                             num_features, num_classes,
#                                             epochs=epochs, lr=lr)

#     plot_results(acc_ebm, loss_ebm, acc_fl, loss_fl, acc_cent, loss_cent, folder="diagram-EBM2")

# if __name__ == "__main__":
#     main()


import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from data_loader import mnistLoader

# ----- Core Functions -----
def hinge_loss(outputs, labels):
    return torch.mean(torch.clamp(1 - outputs.squeeze() * labels, min=0))

def compute_gradients(inputs, outputs, labels):
    margin = 1 - outputs.squeeze() * labels
    grad_mask = (margin > 0).float()
    grad_outputs = -labels * grad_mask
    grad = torch.matmul(inputs.T, grad_outputs.unsqueeze(1)) / inputs.size(0)
    return grad

def evaluate_model(weights, test_loader):
    correct, total = 0, 0
    for inputs, targets in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        logits = torch.matmul(inputs, weights).squeeze()
        preds = (logits > 0).long()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ----- Centralized SVM Training -----
def centralized_svm_sklearn(train_loader, test_loader):
    X_train = train_loader.dataset.data.view(len(train_loader.dataset), -1).numpy() / 255.0
    y_train = (train_loader.dataset.targets.numpy() % 2)

    X_test = test_loader.dataset.data.view(len(test_loader.dataset), -1).numpy() / 255.0
    y_test = (test_loader.dataset.targets.numpy() % 2)

    model = LinearSVC(max_iter=10000, tol=1e-4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[Centralized SVM (sklearn)] Final Accuracy: {accuracy:.4f}")

    acc_curve = [accuracy] * 32
    loss_curve = [1.0] * 32

    return acc_curve, loss_curve

# ----- Federated Training (True EBM and FedAvg) -----
def train_federated(client_loaders, test_loader, num_features, sigma, lr, epochs, robust=False, num_inner_updates=1):
    global_weights = torch.zeros((num_features, 1))
    acc_curve, loss_curve = [], []
    lambd = 0.01  # Regularization term

    G_prev = torch.zeros_like(global_weights)

    for epoch in range(epochs):
        rho_t = 1 / (epoch + 1) ** 0.7
        gamma_t = 1 / (epoch + 1) ** 0.9

        local_models = []
        total_loss = 0

        for loader in client_loaders:
            w_local = global_weights.clone()

            if robust:
                noise = torch.normal(mean=0, std=sigma, size=w_local.shape)
                w_noisy = w_local + noise
            else:
                w_noisy = w_local.clone()

            w_prev = w_noisy.clone()

            for _ in range(num_inner_updates):
                for inputs, targets in loader:
                    inputs = inputs.view(inputs.size(0), -1)
                    labels = 2 * targets.float() - 1

                    logits = torch.matmul(inputs, w_noisy).squeeze()
                    loss = hinge_loss(logits, labels)
                    grad = compute_gradients(inputs, logits.unsqueeze(1), labels)

                    if robust:
                        G_t = (1 - rho_t) * G_prev + rho_t * grad
                        surrogate_grad = 2 * lambd * (w_noisy - w_prev) + G_t
                        w_star = w_noisy - lr * surrogate_grad
                        w_noisy = w_noisy + gamma_t * (w_star - w_noisy)
                        G_prev = G_t.clone()
                    else:
                        w_noisy = w_noisy - lr * grad

                    total_loss += loss.item()

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
    plt.plot(epochs, acc_cent[:len(epochs)], label="Centralized SVM (sklearn)", marker='^')
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
    try:
        dataset_choice = input("Select dataset (MNIST only for now): ").strip().lower()
        epochs = int(input("Enter number of epochs (default=32): ") or 32)
        batch_size = int(input("Enter batch size (default=64): ") or 64)
        sigma = float(input("Enter communication noise level sigma (default=0.01): ") or 0.01)
        lr = float(input("Enter learning rate (default=0.005): ") or 0.005)
        num_clients = int(input("Enter number of clients (default=5): ") or 5)
        num_inner_updates = int(input("Enter number of inner updates per client (default=1): ") or 1)
    except ValueError:
        print("Invalid input. Using default settings.")
        dataset_choice, num_clients, batch_size, epochs, lr, sigma, num_inner_updates = "mnist", 5, 64, 32, 0.005, 0.01, 1

    if dataset_choice != "mnist":
        raise NotImplementedError("Only MNIST supported for this script.")

    train_loader, test_loader = mnistLoader(batch_size=batch_size)
    train_loader.dataset.targets = train_loader.dataset.targets % 2
    test_loader.dataset.targets = test_loader.dataset.targets % 2
    num_features = 28 * 28

    dataset = list(train_loader.dataset)
    split_size = len(dataset) // num_clients
    client_loaders = [
        torch.utils.data.DataLoader(dataset[i * split_size:(i + 1) * split_size], batch_size=batch_size, shuffle=True)
        for i in range(num_clients)
    ]

    acc_cent, loss_cent = centralized_svm_sklearn(train_loader, test_loader)

    acc_ebm, loss_ebm = train_federated(client_loaders, test_loader,
                                        num_features, sigma, lr,
                                        epochs=epochs, robust=True, num_inner_updates=num_inner_updates)

    acc_fl, loss_fl = train_federated(client_loaders, test_loader,
                                      num_features, sigma, lr,
                                      epochs=epochs, robust=False, num_inner_updates=num_inner_updates)

    plot_results(acc_ebm, loss_ebm, acc_fl, loss_fl, acc_cent, loss_cent, folder="diagram-EBM")

if __name__ == "__main__":
    main()
