"""
Flask 应用入口

图像篡改检测系统后端服务。
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config: dict = None) -> Flask:
    """
    Flask 应用工厂

    参数:
        config: 可选的配置字典

    返回:
        Flask 应用实例
    """
    # 计算前端目录的绝对路径
    # 当前文件: src/backend/app.py
    # 前端目录: src/frontend
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(current_dir, "../frontend")
    
    # 计算输出目录绝对路径 (项目根目录/outputs)
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    output_dir = os.path.join(project_root, "outputs")

    # 初始化 Flask，将静态目录指向前端，URL 路径设为空（实现根目录访问静态资源）
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    
    # 配置输出目录
    app.config["OUTPUT_FOLDER"] = output_dir
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    @app.route("/")
    def index():
        """服务前端入口文件"""
        return app.send_static_file("index.html")


    # 默认配置
    app.config.update({
        "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,   # 最大上传 50MB
        "MODEL_PATH": os.getenv("MODEL_PATH", "models/checkpoints/best_model.pth"),
        "MODEL_TYPE": os.getenv("MODEL_TYPE", "dual_stream"),
        "DEVICE": os.getenv("DEVICE", "auto"),
    })

    if config:
        app.config.update(config)

    # 启用 CORS（允许前端跨域访问）
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 初始化推理服务
    try:
        from src.backend.services.inference import InferenceService
        from src.backend.routes.detect import set_inference_service

        service = InferenceService(
            model_path=app.config["MODEL_PATH"],
            model_type=app.config["MODEL_TYPE"],
            device=app.config["DEVICE"],
        )
        set_inference_service(service)
        logger.info("推理服务初始化完成")

    except Exception as e:
        logger.warning(f"推理服务初始化失败: {e}")
        logger.warning("API 将以降级模式运行")

    # 注册路由蓝图
    from src.backend.routes.detect import detect_bp
    app.register_blueprint(detect_bp)

    # 健康检查端点
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "ok",
            "model_type": app.config["MODEL_TYPE"],
            "device": app.config["DEVICE"],
        })

    # 系统信息端点
    @app.route("/api/info", methods=["GET"])
    def system_info():
        import torch
        return jsonify({
            "project": "基于深度学习的图像篡改检测系统",
            "version": "1.0.0",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })

    logger.info("Flask 应用已创建")
    return app


# 直接运行入口
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
