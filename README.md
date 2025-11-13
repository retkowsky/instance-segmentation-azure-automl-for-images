# Azure AutoML for Images - Instance Segmentation Tutorial

[![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-blue)](https://azure.microsoft.com/services/machine-learning/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Overview

This repository provides a comprehensive end-to-end tutorial for building, training, and deploying instance segmentation models using Azure AutoML for Images. The project demonstrates how to leverage Azure Machine Learning's automated ML capabilities to create computer vision models without extensive manual configuration.

Instance segmentation combines object detection with pixel-level segmentation, allowing you to identify and precisely outline multiple objects in images. This repository walks through the entire pipeline from data preparation to model deployment with an interactive web interface.

## 🌟 Key Features

- **Automated Model Training**: Leverage Azure AutoML to automatically select and optimize the best model architecture
- **End-to-End Pipeline**: Complete workflow from data download to model deployment
- **Interactive Web Interface**: Gradio-based application for real-time model inference
- **Support for Multiple Architectures**: Including Mask R-CNN models
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **Production-Ready Deployment**: Deploy models as managed online endpoints in Azure

## 📚 Notebooks Overview

### 1. Data Preparation (`1 Download images files and labels.ipynb`)

This notebook handles the initial data pipeline:
- **Download and organize image datasets** from various sources
- **Perform exploratory data analysis** on image properties
- **Prepare train/validation/test splits** for model training
- **Visualize data distributions** and sample annotations

Key capabilities:
- Automated dataset download and extraction
- Image metadata analysis (dimensions, color modes, file sizes)
- Annotation format conversion utilities
- Data quality validation

### 2. AutoML Training (`2 AutoML for Instance segmentation.ipynb`)

The core training notebook that:
- **Configures Azure ML workspace** and compute resources
- **Creates MLTable data assets** for training and validation
- **Sets up AutoML experiment** with hyperparameter search spaces
- **Trains multiple model architectures** (Mask R-CNN)
- **Evaluates model performance** with various metrics
- **Registers the best model** in Azure ML Model Registry
- **Deploys model to managed endpoint** for online inference

Technical highlights:
- GPU compute cluster configuration
- Advanced hyperparameter tuning with Bandit Policy
- Support for custom model architectures
- Comprehensive performance metrics (mAP, IoU, etc.)
- MLflow experiment tracking

### 3. Model Inference (`3 Instance segmentation model inferencing.ipynb`)

The deployment and inference notebook featuring:
- **Online endpoint management** for production deployment
- **Real-time model inference** with REST API integration
- **Interactive Gradio web application** for user-friendly predictions
- **Batch processing capabilities** for multiple images
- **Visualization utilities** for segmentation results
- **Performance optimization** for inference speed

Application features:
- Adjustable confidence thresholds
- Support for multiple image formats
- Real-time segmentation visualization
- Detailed prediction summaries
- Example images for quick testing

## 🚀 Getting Started

### Prerequisites

- **Azure Subscription** with Machine Learning service enabled
- **Azure ML Workspace** configured
- **Python 3.10+** environment
- **GPU Compute Cluster** (recommended: Standard_NC24ads_A100_v4 or similar)

## 📖 Documentation & Resources

### Official Documentation
- [Azure ML AutoML for Images](https://learn.microsoft.com/azure/machine-learning/concept-automated-ml#computer-vision-preview)
- [Supported Model Architectures](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models)
- [MLflow Integration Guide](https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow)

### Related Tutorials
- [Azure ML Examples Repository](https://github.com/Azure/azureml-examples)
- [Example](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/automl-standalone-jobs/automl-image-instance-segmentation-task-fridge-items)


**Note**: This project requires an active Azure subscription and may incur costs for compute resources and model hosting. Please review [Azure ML pricing](https://azure.microsoft.com/pricing/details/machine-learning/) before deployment.
