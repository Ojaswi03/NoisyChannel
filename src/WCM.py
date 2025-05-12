import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import mnistLoader, cifar10Loader

# ----- Core functions -----
def softmax(logits):
    exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

def cross_entropy_loss(predictions, targets):
    batch_size = predictions.shape[0]
    log_preds = torch.log(predictions + 1e-9)
    return -torch.sum(targets * log_preds) / batch_size

def compute_gradients(inputs, predictions, targets):
    batch_size = inputs.shape[0]
    grad_logits = predictions - targets
    return torch.matmul(inputs.T, grad_logits) / batch_size

def evaluate_model(weights, test_loader, num_classes):
    correct, total = 0, 0
    for inputs, targets in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        logits = torch.matmul(inputs, weights)
        preds = softmax(logits)
        correct += (torch.argmax(preds, dim=1) == targets).sum().item()
        total += targets.size(0)
    return correct / total

def sample_weight_noise(weight_shape, radius):
    noise = torch.randn(weight_shape)
    return radius * noise / torch.norm(noise)

# ----- Federated Training for WCM -----
def train_wcm(client_loaders, test_loader, num_features, num_classes, sigma_w, epochs, lr, lambd, robust=False):
    w_t = torch.zeros((num_features, num_classes))
    G_prev = torch.zeros_like(w_t)
    acc_curve, loss_curve = [], []

    beta = 0.7
    alpha = 0.9

    for epoch in range(epochs):
        total_loss = 0
        rho_t = 1.0 / (epoch + 1) ** beta
        gamma_t = 1.0 / (epoch + 1) ** alpha
        local_models = []

        for loader in client_loaders:
            w_prev = w_t.clone()
            G_t = G_prev.clone()

            for inputs, targets in loader:
                inputs = inputs.view(inputs.size(0), -1)
                targets_one_hot = torch.eye(num_classes)[targets]

                if robust:
                    delta_w = sample_weight_noise(w_t.shape, sigma_w)
                    w_noisy = w_t + delta_w
                else:
                    w_noisy = w_t

                logits = torch.matmul(inputs, w_noisy)
                preds = softmax(logits)
                loss = cross_entropy_loss(preds, targets_one_hot)
                grad = compute_gradients(inputs, preds, targets_one_hot)

                if robust:
                    G_t = (1 - rho_t) * G_prev + rho_t * grad
                    surrogate_grad = 2 * lambd * (w_t - w_prev) + G_t
                    w_star = w_t - lr * surrogate_grad
                    w_local = w_t + gamma_t * (w_star - w_t)
                    surrogate_loss = rho_t * loss + lambd * torch.norm(w_t - w_prev)**2 + (1 - rho_t) * torch.sum((w_t - w_prev) * G_t)
                    total_loss += surrogate_loss.item()
                else:
                    w_local = w_t - lr * grad
                    total_loss += loss.item()

            local_models.append(w_local)
            G_prev = G_t.clone()

        w_t = torch.stack(local_models).mean(dim=0)
        acc = evaluate_model(w_t, test_loader, num_classes)
        acc_curve.append(acc)
        loss_curve.append(total_loss / len(client_loaders))

        print(f"[{'WCM' if robust else 'FedAvg'}] Epoch {epoch+1}: Loss={loss_curve[-1]:.4f}, Accuracy={acc:.4f}")

    return loss_curve, acc_curve

# ----- Centralized Training -----
def train_centralized(train_loader, test_loader, num_features, num_classes, epochs, lr):
    weights = torch.zeros((num_features, num_classes))
    acc_curve = []
    loss_curve = []

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            targets_one_hot = torch.eye(num_classes)[targets]
            logits = torch.matmul(inputs, weights)
            preds = softmax(logits)
            loss = cross_entropy_loss(preds, targets_one_hot)
            grad = compute_gradients(inputs, preds, targets_one_hot)
            weights -= lr * grad
            total_loss += loss.item()

        acc = evaluate_model(weights, test_loader, num_classes)
        acc_curve.append(acc)
        loss_curve.append(total_loss / len(train_loader))
        print(f"[Centralized] Epoch {epoch+1}: Loss={loss_curve[-1]:.4f}, Accuracy={acc:.4f}")

    return loss_curve, acc_curve

# ----- Save Plots -----
def save_comparison_plots(loss_r, acc_r, loss_c, acc_c, loss_cent, acc_cent, outdir):
    os.makedirs(outdir, exist_ok=True)

    plt.plot(acc_r, label="Proposed (WCM)", marker='s')
    plt.plot(acc_c, label="Conventional FL", marker='^')
    plt.plot(acc_cent, label="Centralized")
    plt.title("Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_comparison.png"), dpi = 300)
    plt.clf()

    plt.semilogy(loss_r, label="Proposed (WCM)", marker='s')
    plt.semilogy(loss_c, label="Conventional FL", marker='^')
    plt.semilogy(loss_cent, label="Centralized")
    plt.title("Loss Comparison (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, which='both', axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_comparison.png"), dpi = 300)
    plt.clf()

# ----- Main -----
def main():
    try:
        dataset_choice = input("Select dataset (MNIST/CIFAR10): ").strip().lower()
        epochs = int(input("Enter number of epochs (default=35): ") or 35)
        batch_size = int(input("Enter batch size (default=64): ") or 64)
        sigma_w = float(input("Enter weight noise sigma (default=2.5): ") or 2.5)
        lr = float(input("Enter learning rate (default=0.05): ") or 0.05)
        num_clients = int(input("Enter number of clients (default=10): ") or 10)
    except ValueError:
        print("Invalid input. Using defaults.")
        dataset_choice, epochs, batch_size, sigma_w, lr, num_clients = "cifar10", 35, 64, 2.5, 0.05, 10

    if dataset_choice == "mnist":
        train_loader, test_loader = mnistLoader(batch_size=batch_size)
        num_features = 28 * 28
        num_classes = 10
    elif dataset_choice == "cifar10":
        train_loader, test_loader = cifar10Loader(batch_size=batch_size)
        num_features = 3 * 32 * 32
        num_classes = 10
    else:
        print("Invalid dataset choice. Defaulting to MNIST.")
        train_loader, test_loader = mnistLoader(batch_size=batch_size)
        num_features = 28 * 28
        num_classes = 10

    dataset = list(train_loader.dataset)
    split_size = len(dataset) // num_clients
    client_loaders = [
        torch.utils.data.DataLoader(dataset[i * split_size:(i + 1) * split_size], batch_size=batch_size, shuffle=True)
        for i in range(num_clients)
    ]

    loss_robust, acc_robust = train_wcm(client_loaders, test_loader, num_features, num_classes,
                                        sigma_w=sigma_w, epochs=epochs, lr=lr, lambd=0.01, robust=True)

    loss_conv, acc_conv = train_wcm(client_loaders, test_loader, num_features, num_classes,
                                    sigma_w=0.0, epochs=epochs, lr=lr, lambd=0.0, robust=False)

    loss_cent, acc_cent = train_centralized(train_loader, test_loader, num_features, num_classes,
                                            epochs=epochs, lr=lr)

    save_comparison_plots(loss_robust, acc_robust, loss_conv, acc_conv, loss_cent, acc_cent, outdir="diagram-WCM1")

if __name__ == "__main__":
    main()
