import torch
from modelC10 import CLIP
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = CLIP()

model.load_state_dict(
    torch.load("./data/miniclip.pth", map_location=device)
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_full = datasets.CIFAR10(
    root="./data/cifar10",
    train=False,
    download=True,
    transform=transform
)

# 如果只想测试前1000张
test_dataset = Subset(test_full, range(1000))

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images, torch.arange(10, device=device))

        pred = logits.argmax(dim=1)
        pred_class = pred

        correct += (pred_class == labels).sum().item()
        total += labels.size(0)

print("correct:", correct)
print("total:", total)
print("accuracy:", correct / total)