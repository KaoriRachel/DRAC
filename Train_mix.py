import os.path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision.models
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import random


def read_images(image_path, label_path):
    images = [[], [], []]
    label_pd = pd.read_csv(label_path)
    for _, row in label_pd.iterrows():
        name, label = row['image name'], row['image quality level']
        label = int(label)
        image_file = os.path.join(image_path, name)
        image = Image.open(image_file)
        images[label].append(image)
    print(len(images[0]), len(images[1]), len(images[2]))
    return images


def mix_up(image1, image2, label1, label2, alpha=0.5):
    lam = lam = np.random.beta(alpha, alpha)
    if label1 == label2:
        mixed_image = Image.blend(image1, image2, alpha)
        return [mixed_image, label1]

    while 0.3 < lam < 0.7:
        lam = np.random.beta(alpha, alpha)
    mixed_image = Image.blend(image1, image2, alpha)
    label = label2 if lam <= 0.3 else label1
    return [mixed_image, label]


def mix_up_gen(images, prob, num):
    image_labels_ = []
    for _ in range(num):
        r1 = np.random.choice([0, 1, 2], p=prob)
        r2 = np.random.choice([0, 1, 2], p=prob)
        image1 = random.choice(images[r1])
        image2 = random.choice(images[r2])
        image_labels_.append(mix_up(image1, image2, r1, r2))
    return image_labels_


def merge_images(images_origin, image_labels_gen):
    image_labels_ = []
    for image in images_origin[0]:
        image_labels_.append([image, 0])
    for image in images_origin[1]:
        image_labels_.append([image, 1])
    for image in images_origin[2]:
        image_labels_.append([image, 2])
    for image_label in image_labels_gen:
        image_labels_.append(image_label)
    return image_labels_


# DataSet applying augmentation
class MyDataSet(Dataset):
    def __init__(self, img_height, img_weight, image_labels):
        self.image_labels = image_labels
        self.img_height, self.img_weight = img_height, img_weight

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image, label = self.image_labels[idx]
        image = image.convert('RGB')

        transform = transforms.Compose(
            [
                transforms.Resize((self.img_height, self.img_weight)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.4),
                transforms.ToTensor()
            ]
        )
        image = transform(image)

        return image, label


# params
img_height, img_weight = 256, 256
mix_up_num = 500
num_classes = 3
batch_size = 32
epochs = 12
weight_decay = 1e-3
probability = [0.55, 0.4, 0.05]
image_path = './Training Set'
label_path = './DRAC2022_ Image Quality Assessment_Training Labels.csv'
model_path = './VGGMix.pth'

images = read_images(image_path, label_path)
new_image_labels = mix_up_gen(images, probability, mix_up_num)

image_labels = merge_images(images, new_image_labels)
train_dataset = MyDataSet(img_height, img_weight, image_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Data OK---------------------------------------------------------------")
model = torchvision.models.vgg16(pretrained=True)
print(model)
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=3, bias=True)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
for param in model.parameters():
    param.requires_grad = False
params = model.classifier[6].parameters()
for param in params:
    param.requires_grad = True

device = torch.device('cuda:0')
model = model.to(device)
model.cuda()

optimizer1 = Adam(model.classifier[6].parameters(), lr=1e-4, weight_decay=weight_decay)
scheduler1 = StepLR(optimizer=optimizer1, step_size=2, gamma=0.8)
CELoss = torch.nn.CrossEntropyLoss()

import time

for epoch in range(epochs):
    print('Epoch:', epoch + 1)
    start_time = time.time()
    losses = 0.0
    pred, label = [], []
    max_accuracy = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        pred = pred + outputs.detach().cpu().numpy().argmax(1).tolist()
        label = label + labels.detach().cpu().numpy().tolist()
        loss = CELoss(outputs, labels)

        loss.backward()
        optimizer1.step()
        #optimizer2.step()
        losses += loss.item()

    scheduler1.step()
    #scheduler2.step()
    end_time = time.time()
    print('time:', end_time - start_time)
    print('avg loss:', losses / 21)
    avg_accuracy = accuracy_score(pred, label)
    print('avg accuracy', avg_accuracy)
    print('avg Kappa', cohen_kappa_score(pred, label, weights='quadratic'))
    print()
    if avg_accuracy > max_accuracy:
        max_accuracy = avg_accuracy
        torch.save(model.state_dict(), model_path)
    # store model
