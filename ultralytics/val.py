from ultralytics import YOLO
import warnings
import argparse
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='runs/detect/train/yolov13-4-CFCM-HDSA-ISM/weights/best.pt', help='loading weights')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='', help='save to project/name')
    parser.add_argument('--name', default='test/yolov13-4-CFCM-HDSA-ISM', help='save to project/name')
    return parser.parse_args()

if __name__ == '__main__':
    args=main()
    # 开始加载模型
    model = YOLO(args.weights)
    # 指定训练参数开始验证
    model.val(device=args.device,
              project=args.project,
              name=args.name,
              split='test')
























