#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for People Counting System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Model settings
DEFAULT_MODEL = "yolov8s.pt"
AVAILABLE_MODELS = [
    "yolov8n.pt",  # Fast
    "yolov8s.pt",  # Balanced (default)
    "yolov8m.pt",  # Accurate
    "yolov8l.pt",  # Very accurate
    "yolov9s.pt",  # YOLOv9
    "yolov9s.engine"  # YOLOv9 TensorRT (quantized)
]

# TensorRT settings
TENSORRT_MODELS = [
    "yolov9s.engine"  # TensorRT engines
]
TENSORRT_PRECISION = "fp16"  # fp32, fp16, int8
TENSORRT_WORKSPACE_SIZE = 1024  # MB

# Detection settings
CLASS_ID = 0  # 'person' class
CONF_THRESHOLD = 0.5

# Tracking settings
TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3

# Web server settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_DEBUG = False

# Video settings
DEFAULT_VIDEO = "Test.mp4"
FPS_TARGET = 30

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_model_path(model_name):
    """Get full path to model file"""
    return MODELS_DIR / model_name

def get_video_path(video_name):
    """Get full path to video file"""
    if video_name.isdigit():
        return int(video_name)  # Camera ID
    return DATA_DIR / video_name

def is_tensorrt_model(model_name):
    """Check if model is TensorRT engine"""
    return model_name.endswith('.engine')

def get_model_type(model_name):
    """Get model type: pytorch, onnx, tensorrt"""
    if model_name.endswith('.pt'):
        return 'pytorch'
    elif model_name.endswith('.onnx'):
        return 'onnx'
    elif model_name.endswith('.engine'):
        return 'tensorrt'
    else:
        return 'unknown'
