 
export CUDA_VISIBLE_DEVICES=3,4
python3 tools/train.py -f ./exps/example/yolox_voc/yolox_voc_ct.py -d 1 -b 80 -o -c /media/Datacenter_storage/ramon_dataset_curations/kidney_radiomics_yolo/weights/yolox_nano.pth --logger tensorboard 
