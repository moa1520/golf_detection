import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device:', device)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


if __name__ == '__main__':
    main()
