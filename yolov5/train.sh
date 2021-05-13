'''
# 导入基础镜像
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# opencv-python 依赖
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx

# docker build . -t pytorch:latest
'''
# yolov5 单机多节点
# node 1
sudo docker run -it --rm --gpus 0 -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility -v /media/zw/DL/ly/workspace/project11/yolov5:/usr/yolov5 -w /usr/yolov5 --network=host --ipc=host pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
python train.py --init_method tcp://192.168.210.100:29500 --world_size 2 --rank 0 --backend gloo --local_rank 0 --img 640 --batch 8 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5l.yaml --weights ''

# node 2
sudo docker run -it --rm --gpus 0 -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility -v /home/algdev/liuy/workspace/project11/yolov5:/usr/yolov5 -w /usr/yolov5 --network=host --ipc=host pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
python train.py --init_method tcp://192.168.101.37:29500 --world_size 2 --rank 0 --backend nccl --local_rank 0 --img 640 --batch 8 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5l.yaml --weights ''
python train.py --init_method tcp://192.168.101.37:29500 --world_size 2 --rank 1 --backend nccl --local_rank 1 --img 640 --batch 8 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5l.yaml --weights ''



python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 train.py --img 640 --batch 8 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5l.yaml --weights ''


# train
python train.py --local_rank -1 --img 640 --batch 16 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5s.yaml --weights ''
python train.py --init_method tcp://192.168.101.37:29500 --world_size 2 --rank 0 --backend nccl --local_rank 0 --img 640 --batch 16 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5l.yaml --weights ''
python train.py --init_method tcp://192.168.101.37:29500 --world_size 2 --rank 1 --backend nccl --local_rank 1 --img 640 --batch 16 --epochs 100 --data ./data/custom.yaml --cfg ./models/yolov5l.yaml --weights ''
tensorboard --bind_all --logdir runs/


# test --cfg ./models/yolov5l.yaml
python test.py --img 640 --batch 16 --data ./data/custom.yaml --weights runs/exp2/weights/best.pt --verbose
python detect.py --weights weights/yolov5s.pt --source inference/images/bus.jpg --img-size 640 --conf-thres 0.25 --iou-thres 0.45 --device 0
python models/export.py --weights weights/yolov5s.pt
python -m onnxsim yolov5s.onnx yolov5s-sim.onnx

# to ncnn
cd /media/zw/DL/ly/software/ncnn/build/tools/onnx
./onnx2ncnn yolov5s-sim.onnx yolov5s.param yolov5s.bin
../ncnnoptimize yolov5s.param yolov5s.bin yolov5s-opt.param yolov5s-opt.bin 1 // fp16
../ncnnoptimize yolov5s.param yolov5s.bin yolov5s-opt.param yolov5s-opt.bin 0 // fp32