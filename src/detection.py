import torch


def detect_box(image_path, model):
    #model = torch.hub.load('./src/yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")

    results = model(image_path).xyxy[0].numpy()

    if(len(results) > 0):
        return results[0]
    return []
