import torch
from algorithms.computer.facenet_model.models.facenet import EnhancedFaceNet

# 加载训练好的PyTorch模型
model = EnhancedFaceNet()
model.load_state_dict(torch.load("checkpoints/facenet_best_model.pth"))
model.eval()

# 生成示例输入张量（与训练输入形状一致）
dummy_input = torch.randn(1, 3, 64, 64)  # [batch, channel, height, width]

# 导出为ONNX格式
torch.onnx.export(
    model,
    dummy_input,
    "checkpoints/facenet_best_model.onnx",  # 输出路径
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}  # 支持动态batch
)

print("模型已导出为 facenet_best_model.onnx")