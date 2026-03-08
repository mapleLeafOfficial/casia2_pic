import torch
import sys

def check_gpu():
    print("="*50)
    print("PyTorch GPU 环境检查工具")
    print("="*50)
    
    # 1. 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 2. 检查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 3. 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        # 4. 获取 GPU 数量
        device_count = torch.cuda.device_count()
        print(f"检测到的 GPU 数量: {device_count}")
        
        # 5. 打印每个 GPU 的型号
        for i in range(device_count):
            print(f"GPU {i} 名称: {torch.cuda.get_device_name(i)}")
        
        # 6. 进行简单的矩阵运算测试 GPU 算力
        print("\n正在启动 GPU 压力测试...")
        try:
            # 创建两个大矩阵并移动到 GPU
            x = torch.randn(5000, 5000).cuda()
            y = torch.randn(5000, 5000).cuda()
            
            # 执行矩阵乘法
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            z = torch.matmul(x, y)
            end_event.record()
            
            torch.cuda.synchronize() # 等待计算完成
            
            print(f"GPU 矩阵运算测试成功！耗时: {start_event.elapsed_time(end_event):.2f} ms")
            print("结论：你的 RTX 4060 已准备就绪，可以开始飞速训练了！")
        except Exception as e:
            print(f"GPU 测试运行失败: {e}")
    else:
        print("\n结论：当前环境【无法】使用 GPU。")
        print("请检查：")
        print("1. 是否安装了正确的 CUDA 版本 PyTorch？")
        print("2. 电脑是否安装了最新的 NVIDIA 显卡驱动？")
    
    print("="*50)

if __name__ == "__main__":
    check_gpu()
