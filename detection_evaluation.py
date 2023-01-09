from src.yolo import pr_score
import torch

model = torch.hub.load('./src/yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")

pr_score(model)
