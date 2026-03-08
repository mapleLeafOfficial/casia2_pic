"""
检测 API 路由

处理 POST /api/detect 接口，接收上传图像并返回检测结果。
"""

import os
import uuid
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file, current_app

logger = logging.getLogger(__name__)

detect_bp = Blueprint("detect", __name__)

# 全局推理服务实例（在 app.py 中初始化后注入）
_inference_service = None


def set_inference_service(service):
    """注入推理服务实例"""
    global _inference_service
    _inference_service = service


@detect_bp.route("/api/detect", methods=["POST"])
def detect_tampering():
    """
    图像篡改检测接口

    请求:
        POST multipart/form-data
        - file: 图像文件 (JPG/PNG)

    响应:
        JSON 包含:
        - is_tampered: 是否篡改
        - confidence: 最高置信度
        - tampered_ratio: 篡改面积比
        - result_id: 结果 ID（用于获取可视化图像）
    """
    if _inference_service is None:
        return jsonify({"error": "推理服务未初始化"}), 503

    # 检查文件
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"不支持的文件格式: {ext}"}), 400

    try:
        # 读取图像
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "无法解码图像文件"}), 400

        # 推理
        result = _inference_service.predict(image)

        # 生成结果 ID
        result_id = str(uuid.uuid4())[:8]

        # 保存结果到输出目录 (使用绝对路径)
        output_dir = Path(current_app.config["OUTPUT_FOLDER"]) / result_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存原图
        cv2.imwrite(str(output_dir / "original.png"), image)

        # 保存二值掩码
        cv2.imwrite(str(output_dir / "mask.png"), result["binary_mask"])

        # 保存概率图（灰度）
        prob_uint8 = (result["probability_map"] * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "probability.png"), prob_uint8)

        # 保存叠加可视化
        from src.backend.utils.visualization import overlay_mask, overlay_heatmap
        overlay = overlay_mask(image, result["binary_mask"])
        cv2.imwrite(str(output_dir / "overlay.png"), overlay)

        heatmap = overlay_heatmap(image, result["probability_map"])
        cv2.imwrite(str(output_dir / "heatmap.png"), heatmap)

        # 生成检测报告
        from src.backend.utils.visualization import generate_detection_report
        report = generate_detection_report(result["probability_map"])

        return jsonify({
            "result_id": result_id,
            "is_tampered": result["is_tampered"],
            "confidence": round(result["confidence"], 4),
            "tampered_ratio": round(result["tampered_ratio"], 4),
            "num_regions": report["num_regions"],
            "regions": report["regions"],
        })

    except Exception as e:
        logger.exception("检测过程出错")
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


@detect_bp.route("/api/result/<result_id>/<image_type>", methods=["GET"])
def get_result_image(result_id: str, image_type: str):
    """
    获取检测结果图像

    路径参数:
        - result_id: 检测结果 ID
        - image_type: 图像类型 (original, mask, probability, overlay, heatmap)
    """
    valid_types = {"original", "mask", "probability", "overlay", "heatmap"}
    if image_type not in valid_types:
        return jsonify({"error": f"无效的图像类型: {image_type}"}), 400

    image_path = Path(current_app.config["OUTPUT_FOLDER"]) / result_id / f"{image_type}.png"
    if not image_path.exists():
        return jsonify({"error": "结果不存在"}), 404

    return send_file(str(image_path), mimetype="image/png")


@detect_bp.route("/api/batch-detect", methods=["POST"])
def batch_detect():
    """
    批量图像篡改检测接口

    请求:
        POST multipart/form-data
        - files: 多个图像文件

    响应:
        JSON 数组，每个元素包含单张图像的检测结果
    """
    if _inference_service is None:
        return jsonify({"error": "推理服务未初始化"}), 503

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "未上传文件"}), 400

    results = []
    for file in files:
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "无法解码图像",
                })
                continue

            result = _inference_service.predict(image)
            results.append({
                "filename": file.filename,
                "is_tampered": result["is_tampered"],
                "confidence": round(result["confidence"], 4),
                "tampered_ratio": round(result["tampered_ratio"], 4),
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })

    return jsonify(results)
