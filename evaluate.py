from detection_model import MultiModalYOLO

print("Loading model...")
detector = MultiModalYOLO(model_size='s')
detector.load('runs/detect/multimodal_yolo/weights/best.pt')

print("\nEvaluating on validation set...")
metrics = detector.evaluate('processed_data/dataset.yaml')

print(f'\nResults:')
print(f'  mAP@0.5:      {metrics.box.map50:.4f}')
print(f'  mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'  Precision:    {metrics.box.mp:.4f}')
print(f'  Recall:       {metrics.box.mr:.4f}')
