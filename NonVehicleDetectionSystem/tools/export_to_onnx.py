"""
Export a trained PyTorch model to ONNX.
Usage:
  python export_to_onnx.py --checkpoint path/to/model.pth --output model/vehicle_detector.onnx --num-classes 2

This is adapted from the notebook export cell (uses input size 224x224, input name 'input', output name 'output').
"""
import argparse
import torch
import torchvision
from torch import nn


def build_model(num_classes: int):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', default='model/vehicle_detector.onnx', help='Output ONNX file')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    args = parser.parse_args()

    model = build_model(args.num_classes)
    print('Loading checkpoint: ', args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu')

    # Attempt to support both state_dict and full model saves
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        try:
            model.load_state_dict(state)
        except Exception as e:
            print('Warning: failed to load state directly:', e)
            # If the checkpoint is a full model, try replacing model
            model = state

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print('ONNX exported to', args.output)
