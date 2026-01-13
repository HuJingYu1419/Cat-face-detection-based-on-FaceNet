import torch
from torch import nn
from torch.utils.data import DataLoader
from models.facenet_pi import LiteFaceNet  # 使用轻量化模型
from utils.losses import TripletLoss
from utils.dataloader import CatTripletDataset, split_dataset
from torch.utils.tensorboard import SummaryWriter
import os
from torch.cuda.amp import GradScaler, autocast


def generate_triplets(full_dataset, subset, batch_size, batch_idx, batch_data):
    anchors, positives, negatives = [], [], []
    for i in range(len(batch_data[0])):
        global_idx = subset.indices[batch_idx * batch_size + i]
        a, p, n = full_dataset.get_triplet(global_idx)
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def main():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LiteFaceNet(embedding_size=32).to(device)  # 更小的嵌入维度
    criterion = TripletLoss(margin=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())  # 混合精度训练
    writer = SummaryWriter('runs/experiment_lite')

    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')
    save_every_n_epochs = 2

    # 数据加载
    full_dataset = CatTripletDataset('data/')
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 训练循环
    for epoch in range(10):
        model.train()
        train_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            anchor, positive, negative = generate_triplets(
                full_dataset, train_dataset, train_loader.batch_size, batch_idx, batch_data)

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # 混合精度训练
            with autocast():
                emb_anchor = model(anchor)
                emb_pos = model(positive)
                emb_neg = model(negative)
                loss = criterion(emb_anchor, emb_pos, emb_neg)

            train_loss += loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch:02d} | Train Batch: {batch_idx:03d} | Loss: {loss.item():.4f}')

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, val_batch_data in enumerate(val_loader):
                anchor, positive, negative = generate_triplets(
                    full_dataset, val_dataset, val_loader.batch_size, val_batch_idx, val_batch_data)

                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                loss = criterion(model(anchor), model(positive), model(negative))
                val_loss += loss.item()

        # 记录日志
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)

        # 保存最佳模型
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model_lite.pth')
            print(f'Best model saved. Val Loss: {best_loss:.4f}')

        # 导出ONNX模型
        if epoch % save_every_n_epochs == 0:
            model.eval()
            dummy_input = torch.randn(1, 1, 48, 48).to(device)  # 灰度输入48x48
            torch.onnx.export(
                model,
                dummy_input,
                f'checkpoints/model_lite_epoch{epoch}.onnx',
                export_params=True,
                opset_version=12,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f'ONNX model saved at epoch {epoch}')

    # 动态量化
    print("Starting dynamic quantization...")
    model.eval()

    # 方案1：尝试轻量级动态量化
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # 仅量化全连接层
            dtype=torch.qint8
        )
        # 保存状态字典
        torch.save({
            'state_dict': quantized_model.state_dict(),
            'model_type': 'dynamic_quantized'
        }, 'checkpoints/quantized_model_lite.pth')
        print("Dynamic quantized model (state_dict) saved successfully")
    except Exception as e:
        print(f"Dynamic quantization failed: {str(e)}")
        print("Falling back to non-quantized model")
        torch.save({
            'state_dict': model.state_dict(),
            'model_type': 'original'
        }, 'checkpoints/non_quantized_model_lite.pth')


if __name__ == "__main__":
    main()