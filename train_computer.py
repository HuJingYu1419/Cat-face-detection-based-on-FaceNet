import torch
from torch.utils.data import DataLoader
from algorithms.computer.facenet_model.models.facenet import EnhancedFaceNet
from algorithms.computer.facenet_model.utils.losses import TripletLoss
from algorithms.computer.facenet_model.utils.dataloader import CatTripletDataset, split_dataset
import os
import matplotlib.pyplot as plt
import numpy as np


def generate_triplets(full_dataset, subset, batch_size, batch_idx, batch_data):
    anchors, positives, negatives = [], [], []
    for i in range(len(batch_data[0])):  # Use actual batch size (might be smaller for last batch)
        # Get the global index from the subset indices
        global_idx = subset.indices[batch_idx * batch_size + i]  # 直接使用subset的索引映射
        a, p, n = full_dataset.get_triplet(global_idx)
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def plot_training_curves(train_losses, val_losses, save_path='checkpoints/facenet_training_curves.png'):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 图片保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失曲线
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    
    # 绘制验证损失曲线
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    # 添加标记点
    min_val_loss_idx = np.argmin(val_losses)
    plt.scatter(min_val_loss_idx + 1, val_losses[min_val_loss_idx], 
                color='red', s=100, zorder=5, 
                label=f'Best Val Loss: {val_losses[min_val_loss_idx]:.4f}')
    
    # 设置图形属性
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 设置x轴为整数刻度
    plt.xticks(epochs)
    
    # 添加背景色区分
    plt.axvspan(min_val_loss_idx + 0.5, min_val_loss_idx + 1.5, 
                alpha=0.1, color='green', label='Best Model Epoch')
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    
    # 关闭图形释放内存
    plt.close()


def main():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedFaceNet().to(device)
    criterion = TripletLoss(margin=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)

    # 新增变量：跟踪最佳损失和保存频率
    best_loss = float('inf')
    save_every_n_epochs = 5  # 每n个epoch保存一次模型(自定义周期性保存)
    
    # 用于记录损失的列表
    train_loss_history = []
    val_loss_history = []

    # 数据加载与划分
    full_dataset = CatTripletDataset('data/facenet_dataset', img_size=64)
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # 验证集不需shuffle
    
    # 参数设置
    num_epochs = 3 # 训练轮数
    
    print("=" * 60)
    print("Starting Training...")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Total epochs: {num_epochs}")
    print("=" * 60)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        #训练阶段
        for batch_idx, batch_data in enumerate(train_loader):
            # 在线生成Triplet
            # 训练调用
            anchor, positive, negative = generate_triplets(full_dataset, train_dataset, train_loader.batch_size, batch_idx, batch_data)

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # 前向计算
            emb_anchor = model(anchor)
            emb_pos = model(positive)
            emb_neg = model(negative)

            # 计算损失
            loss = criterion(emb_anchor, emb_pos, emb_neg)
            train_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch:02d} | Train Batch: {batch_idx:03d} | Loss: {loss.item():.4f}')

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, val_batch_data in enumerate(val_loader):
                # 验证调用
                anchor, positive, negative = generate_triplets(full_dataset, val_dataset, val_loader.batch_size, val_batch_idx, val_batch_data)
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                # 计算验证损失
                loss = criterion(model(anchor), model(positive), model(negative))
                val_loss += loss.item()

                if val_batch_idx % 100 == 0:
                    print(f'Epoch: {epoch:02d} | Val Batch: {val_batch_idx:03d} | Loss: {loss.item():.4f}')

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # 打印epoch总结
        print("-" * 50)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} Summary:")
        print(f"  Training Loss:   {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)

        # 根据验证损失保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/facenet_best_model.pth')
            print(f'✓ Best model saved. Val Loss: {best_loss:.4f}')

        # 周期性保存模型
        if epoch % save_every_n_epochs == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch{epoch}.pth')
            print(f'✓ Model saved at epoch {epoch}')
            
        # 每1个epoch绘制一次曲线（或者可以调整频率）
        if (epoch + 1) % 1 == 0 or epoch == num_epochs - 1:
            plot_training_curves(train_loss_history, val_loss_history)
    
    # 训练结束后保存最终模型
    torch.save(model.state_dict(), 'checkpoints/facenet_final_model.pth')
    print(f'✓ Final model saved to checkpoints/facenet_final_model.pth')
    
    # 绘制最终的训练曲线
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    plot_training_curves(train_loss_history, val_loss_history)
    
    # 打印训练统计信息
    print("\nTraining Statistics:")
    print(f"  Best Validation Loss: {min(val_loss_history):.4f} (Epoch {np.argmin(val_loss_history) + 1})")
    print(f"  Final Training Loss: {train_loss_history[-1]:.4f}")
    print(f"  Final Validation Loss: {val_loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()