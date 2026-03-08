import os
import torch
from torchvision import models

def setup_directories():
    """创建必要的模型存放目录"""
    dirs = ['models/checkpoints', 'models/exported']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"创建目录: {d}")

def get_backbone_weights():
    """
    初始化 ResNet-18 骨干权重。
    PyTorch 会自动下载到 ~/.cache/torch/hub/checkpoints/
    这里只是演示如何手动管理。
    """
    print("\n--- 骨干网络初始化 ---")
    print("正在加载 ResNet-18 ImageNet 预训练权重...")
    try:
        # 这会触发自动下载（如果本地没有）
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print("成功加载 ResNet-18 权重。")
    except Exception as e:
        print(f"权重加载失败: {e}")

def list_sota_resources():
    """提供行业领先模型的参考资源"""
    print("\n" + "="*50)
    print(" 推荐的现成图像篡改检测模型 (SOTA)")
    print("="*50)
    print("由于模型文件较大，建议手动从以下官方仓库下载权重后放入 models/checkpoints/：")
    
    resources = [
        {
            "name": "ManTra-Net",
            "desc": "基于异常检测的通用篡改定位模型",
            "url": "https://github.com/stevenwu94/ManTraNet"
        },
        {
            "name": "MVSS-Net",
            "desc": "多视图语义分割网络，性能极佳",
            "url": "https://github.com/dong03/MVSS-Net"
        },
        {
            "name": "CAT-Net",
            "desc": "利用压缩伪影的高分辨率篡改检测",
            "url": "https://github.com/mjchoi01/CAT-Net"
        }
    ]
    
    for res in resources:
        print(f"\n模型名称: {res['name']}")
        print(f"功能描述: {res['desc']}")
        print(f"项目链接: {res['url']}")
    print("\n" + "="*50)

if __name__ == "__main__":
    setup_directories()
    get_backbone_weights()
    list_sota_resources()
    print("\n提示: 如果你有自己训练好的 .pth 文件，请将其重命名为 'best_model.pth' 并放入 models/checkpoints/ 即可通过后端加载。")
