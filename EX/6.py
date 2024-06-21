import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from macrograd.nn import MLP

# 下載並準備 MNIST 數據集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 定義神經網絡模型
model = MLP(nin=784, nouts=[128, 64, 10], act='relu')

# 自定義交叉熵損失函數
def cross_entropy_loss(outputs, targets):
    m = targets.shape[0]
    p = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
    log_likelihood = -np.log(p[range(m), targets])
    loss = np.sum(log_likelihood) / m
    return loss

# 訓練模型函數
def train(model, trainloader, epochs=5, learning_rate=0.01):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, 28*28).numpy()
            labels = labels.numpy()

            # 前向傳播
            outputs = model(inputs)
            
            # 計算損失
            loss = cross_entropy_loss(outputs, labels)
            running_loss += loss

            # 反向傳播和優化
            model.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.data -= learning_rate * param.grad

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

# 評估模型函數
def evaluate(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(-1, 28*28).numpy()
            labels = labels.numpy()
            outputs = model(images)
            predicted = np.argmax(outputs, axis=1)
            total += labels.size
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# 訓練模型
train(model, trainloader)

# 評估模型
evaluate(model, testloader)
