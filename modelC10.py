import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from torchvision import datasets, transforms
from torch.utils.data import random_split

import random
from torch.utils.data import Sampler
from torch.utils.data import Subset

import time

class S_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        self.signals_feat = resnet
        self.w_i = nn.Parameter(torch.rand(512, 256))

        for p in self.signals_feat.parameters():
            p.requires_grad = False


    def forward(self,x):
        I_f = self.signals_feat(x)
        I_e = I_f @ self.w_i
        I_e = F.normalize(I_e, p=2, dim=1)

        return I_e

class T_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(1, 2)
        cache_path = "./data/hf_cache"

        self.token_feat = AutoModel.from_pretrained(
            "distilbert-base-multilingual-cased", cache_dir=cache_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-multilingual-cased", cache_dir=cache_path
        )

        self.w_i = nn.Parameter(torch.rand(768, 256))

        for p in self.token_feat.parameters():
            p.requires_grad = False

    def tensor2strlist(self, labels):
        labels_map = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]
        strlist = [labels_map[int(value.item())//1] for value in labels]
        return strlist

    def forward(self,x):
        if torch.is_tensor(x):
            label_str_list = self.tensor2strlist(x)
        else:
            label_str_list = x
        inputs = self.tokenizer(
            label_str_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        device = next(self.token_feat.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        T_f = self.token_feat(**inputs).last_hidden_state
        T_f = T_f[:,0]

        T_e = T_f @ self.w_i
        T_e = F.normalize(T_e, p=2, dim=1)

        # print(T_e.shape)

        return T_e

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_encoder = S_Encoder()
        self.t_encoder = T_Encoder()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, labels):
        I_e = self.s_encoder(x)
        T_e = self.t_encoder(labels)

        logits = logits = self.logit_scale.exp()*(I_e @ T_e.T)
        return logits

    def loss(self, logits, labels):
        num_classes = logits.shape[1]
        labels = torch.tensor(list(range(num_classes)), dtype=torch.long)

        loss_t = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.transpose(0, 1), labels)
        loss = (loss_i + loss_t)/2

        return loss


class UniqueClassBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.class_to_indices = {}

        for subset_index, original_index in enumerate(dataset.indices):
            label = dataset.dataset.targets[original_index]

            if label not in self.class_to_indices:
                self.class_to_indices[label] = []

            self.class_to_indices[label].append(subset_index)

    def __iter__(self):
        classes = list(self.class_to_indices.keys())
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            selected_classes = random.sample(classes, self.batch_size)

            batch = []

            for cls in selected_classes:
                index = random.choice(self.class_to_indices[cls])
                batch.append(index)

            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

# x = torch.rand(32,3,640,640)
# label = torch.randint(0,10,(32,1))
# model = CLIP()
# print(model(x, label).shape)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_full = datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        transform=transform
    )

    test_full = datasets.CIFAR10(
        root="./data/cifar10",
        train=False,
        download=True,
        transform=transform
    )

    train_dataset = Subset(train_full, range(5000))
    test_dataset = Subset(test_full, range(1000))

    train_sampler = UniqueClassBatchSampler(train_dataset, batch_size=10)
    test_sampler = UniqueClassBatchSampler(test_dataset, batch_size=10)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
    )


    start_time = time.time()
    model = CLIP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 10

    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images, labels)
            loss = model.loss(logits, labels)

            loss.backward()
            optimizer.step()

            print(loss.item())
        elapsed = time.time() - start_time
        print(f"已经运行: {elapsed / 60:.1f} 分钟")


    torch.save(model.state_dict(), "./data/miniclip.pth")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            # 得到 CLIP 相似度矩阵
            logits = model(images, labels)

            # 每张图片最匹配第几个文本
            pred = logits.argmax(dim=1)

            # 文本位置 → 真正的类别编号
            pred_class = labels[pred]

            # 统计正确数量
            correct += (pred_class == labels).sum().item()
            total += labels.size(0)

            print("pred: ", pred_class)
            print("label:", labels)

    accuracy = correct / total

    print("correct:", correct)
    print("total:", total)
    print("accuracy:", accuracy)
