from utils.load_dataset import load_dataset
import torch
from torch import nn
from torchvision import transforms
from timm.models import vit_base_patch16_224
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def computeAUROC(dataPRED, dataGT, classCount=14):

    outAUROC = []
    fprs, tprs, thresholds = [], [], []
    
    for i in range(classCount):
        try:
            # pred_probs = dataPRED[:, i]
            pred_probs = torch.sigmoid(torch.tensor(dataPRED[:, i]))
            fpr, tpr, threshold = roc_curve(dataGT[:, i], pred_probs)
           
            roc_auc = roc_auc_score(dataGT[:, i], pred_probs)
            outAUROC.append(roc_auc)

            # Store FPR, TPR, and thresholds for each class
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)
        except:
            outAUROC.append(0.)

    auc_each_class_array = np.array(outAUROC)

    print("each class: ",auc_each_class_array)
    # Average over all classes
    result = np.average(auc_each_class_array[auc_each_class_array != 0])
    # print(result)
    plt.figure(figsize=(10, 8))  # Đặt kích thước hình ảnh chung

    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], label=f'Class {i} (AUC = {outAUROC[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for all Classes')
    plt.legend()

    output_file = f'./roc_auc.png'  # Đường dẫn lưu ảnh

    # Lưu hình xuống file
    plt.savefig(output_file)

    return result


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
    parser.add_argument('--crop', dest='crop', type=int, default=224,
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)  # Chuyển dữ liệu lên GPU
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Average Loss: {average_loss:.4f}")

        model.eval()
        val_pred_probs, val_true_labels = [], []

        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_images, val_labels = val_batch
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                val_pred_probs.extend(val_outputs.cpu().numpy())
                val_true_labels.extend(val_labels.cpu().numpy())

        val_pred_probs = np.array(val_pred_probs)
        val_true_labels = np.array(val_true_labels)

        # Compute ROC AUC on validation set
        val_auc = computeAUROC(val_pred_probs, val_true_labels)
        print(f"Epoch {epoch + 1}: Validation AUC: {val_auc:.4f}")
    # Lưu mô hình
    torch.save(model.state_dict(), "vit_multilabel.pt")

if __name__=='__main__':
    main()
