import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# 对比
# 1、模型结构调整：3层卷积+池化 和 2层卷积＋池化
# 2、Adam和RMSprop优化器的学习率调整：0.01 和 0.001
# 3、损失函数调整：交叉熵损失、带标签平滑的交叉熵（Label Smoothing CrossEntropy），至于MSE，则不推荐

print(torch.__file__)
print(torch.cuda.is_available())
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

# 计算得到的Fashion-MNIST的均值和方差
mean = 0.286
std = 0.353
#1、 数据集下载和预处理增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # 衣服左右对称，可以翻转
    transforms.RandomRotation(degrees=10), # 随机旋转±10度
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,)) # 归一化
])
# 测试集不增强只归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
# 直接下载
mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=train_transform, download = True)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=test_transform, download = True)

print(len(mnist_train), len(mnist_test))
print(len(mnist_test[0]))

# 先下载再读入
# DATA_PATH=Path('./data/')
# train = pd.read_csv(DATA_PATH / "fashion-mnist_train.csv");
# print(train.head())

#2、 数据加载
batch_size = 128 # 训练批次大小，每次训练输入128张图片
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

#3、定义基准模型（简单CNN）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential( # 特征提取网络：三层卷积+激活+池化
            nn.Conv2d(1, 64, kernel_size=3, padding=1), # 灰度图的输入通道为1，输出通道64，大小28*28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 经过第一次池化-> 14*14
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 输入通道为64，输出通道128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 第二次池化-> 7*7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 第三次池化-> 3*3 7/2=3.5向下取整
        )
        self.classifier = nn.Sequential( # 分类器：两个线性层，先升维再降维
            nn.Linear(256 * 3 * 3, 512), # 第二次池化后大小为128*7*7,作为全连接层的输入
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 神经元随机失活，防止过拟合
            nn.Linear(512, 10)  # Fashion-mnist有10个类别
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        return x

# 4、定义训练和验证函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # 使用 tqdm 创建进度条，并设置描述和颜色
    with tqdm(total=len(train_loader), desc='Training', unit='batch', colour='red') as pbar:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 清零梯度

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{running_loss / (batch_idx + 1):.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

#5、 评估训练结果
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(), tqdm(total=len(test_loader), desc='Validation', unit='batch', colour='red') as pbar:
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{running_loss/(batch_idx+1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

    val_loss = running_loss / len(test_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


# 6. 定义优化器对比实验
def run_optimizer_comparison(optimizers, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用GPU进行训练
    print(f"Using device: {device}")

    results = {}  # 存储每个优化器的结果

    for name, optimizer_fn in optimizers.items():
        print(f"\n----- Training with {name} -----")
        # 初始化模型、损失函数和优化器
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()  # 定义损失函数--交叉熵损失
        optimizer = optimizer_fn(model.parameters())  # 初始化优化器

        # 记录训练过程
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        times = []

        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, test_loader, criterion, device)

            # 记录指标
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            times.append(time.time() - epoch_start)

            # 训练过程输出：进度、准确率、损失、耗费时间
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"Time: {times[-1]:.2f}s")
            # print("\n")

        total_time = time.time() - start_time  # 训练一共耗费的时间
        print(f"Total training time: {total_time:.2f}s")

        # 保存结果
        results[name] = {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'time_per_epoch': times,
            'total_time': total_time
        }

    return results

# 7. 结果可视化
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results, metric='loss', save_path=None):
    # 使用 seaborn 的美化风格
    sns.set(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # 定义颜色（确保每个优化器有唯一且美观的颜色）
    colors = sns.color_palette("Set1", n_colors=len(results))

    for idx, (name, data) in enumerate(results.items()):
        color = colors[idx]
        if metric == 'loss':
            ax.plot(data['train_loss'], label=f'{name} Train', color=color, linestyle='-')
            ax.plot(data['val_loss'], label=f'{name} Val', color=color, linestyle='--')
            ax.set_ylabel('Loss')
        elif metric == 'acc':
            ax.plot(data['train_acc'], label=f'{name} Train', color=color, linestyle='-')
            ax.plot(data['val_acc'], label=f'{name} Val', color=color, linestyle='--')
            ax.set_ylabel('Accuracy (%)')

    ax.set_xlabel('Epoch')
    title = f'Optimizer Comparison – {metric.capitalize()}'
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

if __name__=='__main__':
    # 定义待对比的优化器（相同学习率，其他参数用默认值）
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),  # 带动量的SGD
        'Adam': lambda params: optim.Adam(params, lr=0.001),
        'RMSprop': lambda params: optim.RMSprop(params, lr=0.001),  # 若要lr=0.01,则要将eps设为1e-3或更大
        'Adagrad': lambda params: optim.Adagrad(params, lr=0.01)
    }

    # 运行实验（可根据需求调整epoch数）
    num_epochs = 10
    results = run_optimizer_comparison(optimizers, num_epochs)

    # 绘制损失对比图
    plot_results(results, metric='loss')

    # 绘制准确率对比图
    plot_results(results, metric='acc')

    # 打印最终验证准确率和总训练时间
    print("\nFinal Results:")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Final Val Acc: {data['val_acc'][-1]:.2f}%")
        print(f"  Total Training Time: {data['total_time']:.2f}s")




