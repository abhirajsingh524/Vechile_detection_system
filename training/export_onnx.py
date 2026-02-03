import torch
from torchvision import models
import argparse
from pathlib import Path


def export_to_onnx(pth_path, classes, out_path, img_size=224):
    device = torch.device('cpu')
    num_classes = len(classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    data = torch.load(pth_path, map_location=device)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out_path), input_names=['input'], output_names=['output'], opset_version=11)
    print('Exported ONNX to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='path to .pth model')
    parser.add_argument('--classes', required=True, help='comma separated class names')
    parser.add_argument('--out', default='NonVehicleDetectionSystem/model/vehicle_detector.onnx')
    args = parser.parse_args()

    cls = [c.strip() for c in args.classes.split(',')]
    export_to_onnx(args.pth, cls, args.out)
