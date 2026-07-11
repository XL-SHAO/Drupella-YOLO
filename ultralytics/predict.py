from ultralytics import YOLO
# engineresults
# if pred_boxes and show_boxes:
import warnings
import argparse
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='runs/detect/train/yolov13n/weights/best.pt', help='loading weights')
    parser.add_argument('--datasets',type=str, default='datasets/drupella_dataset/images/test', help='loading weights')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='', help='save to project/name')
    parser.add_argument('--name', default='predict/yolov13n', help='save to project/name')
    return parser.parse_args()

def predictmodel(args):
      # 开始加载模型
      model = YOLO(args.weights)
      # 指定训练参数开始测试
      for i in model.predict(source=args.datasets, 
                             stream=True, 
                             conf=0.3,#15
                             iou=0.55,#55
                             device=args.device,
                             project=args.project, 
                             name=args.name, 
                             save_txt=True, 
                             save=True):
            #print(i)
            pass

if __name__ == "__main__":
      args=main()
      # 调用测试方法
      predictmodel(args)

























































