import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = load_dataset("Bingsu/Cat_and_Dog", split='train')
print(dataset)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["labels"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}


dataset = dataset.with_transform(transforms)


dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=32, shuffle=True, pin_memory=True)

for batch, (X, y) in enumerate(dataloader):
    print(X.shape)


    # Forward pass, loss calculation, and optimization steps here
