python train.py --epochs 100 --batch 16 --cfg 'cfg/yolo-fastest.cfg' --data 'data/custom.data' --multi --img-size 608 --weights ''
python detect.py --cfg 'cfg/yolo-fastest.cfg' --names 'data/custom.names' --img-size 608 --weights 'weights/best.pt'
