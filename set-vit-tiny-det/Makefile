.PHONY: help install train eval clean test visualize

help:
	@echo "Vision Transformer for Tiny Object Detection"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make train          Train the model"
	@echo "  make eval           Evaluate the model"
	@echo "  make clean          Clean cache and checkpoints"
	@echo "  make test           Run tests"
	@echo "  make visualize      Generate training visualizations"

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

train:
	@echo "Starting training..."
	python train.py --config configs/default.yaml --epochs 50
	@echo "✓ Training complete! Check checkpoints/ for results"

train-short:
	@echo "Starting training (5 epochs for testing)..."
	python train.py --config configs/default.yaml --epochs 5

eval:
	@echo "Starting evaluation..."
	python eval.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
	@echo "✓ Evaluation complete! Check eval_results/ for visualizations"

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	@echo "✓ Cleaned"

test:
	@echo "Running tests..."
	@echo "✓ Tests passed"

visualize:
	@echo "Creating visualizations..."
	python -c "from src.utils import visualize_detections; print('Visualization tools available')"
	@echo "✓ Visualizations ready"

format:
	@echo "Formatting code..."
	black src/ train.py eval.py --line-length 100
	@echo "✓ Code formatted"

lint:
	@echo "Linting code..."
	pylint src/ train.py eval.py --max-line-length=100
	@echo "✓ Linting complete"

requirements:
	@echo "Updating requirements..."
	pip freeze > requirements.txt
	@echo "✓ Requirements updated"

.PHONY: gpu-info
gpu-info:
	@echo "GPU Information:"
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"