import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------- Core Functions from Scratch -----------

# Softmax function
def softmax(logits):
    exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

# Cross-entropy loss (mean over batch)
def cross_entropy_loss(predictions, targets):
    batch_size = predictions.shape[0]
    log_preds = torch.log(predictions + 1e-9)  # numerical stability
    loss = -torch.sum(targets * log_preds) / batch_size
    return loss

# Gradient of cross-entropy loss w.r.t weights
def compute_gradients(inputs, predictions, targets):
    batch_size = inputs.shape[0]
    grad_logits = predictions - targets  # derivative w.r.t logits
    grad_weights = torch.matmul(inputs.T, grad_logits) / batch_size  # (features x classes)
    return grad_weights

# Gradient norm squared ( ||grad||^2 )
def gradient_norm_squared(grad):
    return torch.sum(grad**2)

# SGD update step
def sgd_update(weights, grad, lr):
    return weights - lr * grad

# ----------- Expectation-Based Model Training -----------

def train_expectation_based_model(train_loader, num_features, num_classes, sigma_e=0.1, epochs=5, lr=0.1):
    # Initialize weights
    weights = torch.zeros((num_features, num_classes), requires_grad=False)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten (batch_size x num_features)
            targets_one_hot = torch.eye(num_classes)[targets]  # One-hot encoding

            logits = torch.matmul(inputs, weights)
            preds = softmax(logits)
            loss = cross_entropy_loss(preds, targets_one_hot)

            grad = compute_gradients(inputs, preds, targets_one_hot)
            grad_norm_sq = gradient_norm_squared(grad)

            # Robust Loss: L + σ_e^2 ||grad||^2
            robust_loss = loss + sigma_e**2 * grad_norm_sq

            # Robust Gradient: grad + 2σ_e^2 * grad
            robust_grad = grad + 2 * sigma_e**2 * grad

            # Update weights
            weights = sgd_update(weights, robust_grad, lr)

            total_loss += robust_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Robust Loss: {avg_loss:.4f}")
    
    return weights

# ----------- Evaluation -----------

def evaluate_model(weights, test_loader, num_classes):
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        logits = torch.matmul(inputs, weights)
        preds = softmax(logits)
        predicted_classes = torch.argmax(preds, dim=1)
        correct += (predicted_classes == targets).sum().item()
        total += targets.size(0)
    return correct / total

# ----------- Main -----------

def main():
    # DataLoader setup
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    num_features = 28 * 28
    num_classes = 10

    weights = train_expectation_based_model(train_loader, num_features, num_classes, sigma_e=0.1, epochs=5, lr=0.1)
    accuracy = evaluate_model(weights, test_loader, num_classes)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
