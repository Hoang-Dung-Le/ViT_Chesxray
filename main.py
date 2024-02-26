from utils.load_dataset import load_dataset
import torch
from torch import nn
from torchvision import transforms
from timm.models import vit_base_patch16_224
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2', 'chexpert'],
                    help='version of data augmentation to use')
    parser.add_argument('--maintain-ratio', dest='maintain_ratio', action='store_true',
                    help='whether to maintain aspect ratio or scale the image')
    parser.add_argument('--rotate', dest='rotate', action='store_true',
                    help='to rotate image')
    parser.add_argument('--crop', dest='crop', type=int, default=299,
                    help='image crop (Chexpert=320)')

    parser.add_argument('--num_classes', dest='num_classes', type=int, default=14,
                        help='Number of classes')

    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                            help='dataset path')

    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")


    args = parser.parse_args()

    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs

    
    model = vit_base_patch16_224(num_classes=num_classes)

    train_dataset = load_dataset(split='train', args=args)
    val_dataset = load_dataset(split='val', args=args)
    # test_dataset = load_dataset(split='test', args=args)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )


    # Huấn luyện mô hình
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}: Loss: {loss.item()}")

    # Lưu mô hình
    torch.save(model.state_dict(), "vit_multilabel.pt")

if __name__=='__main__':
    main()
