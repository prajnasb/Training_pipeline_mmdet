import os

from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result, init_detector

checkpoint_file = "/media/preeth/Data/prajna_files/mmdet_local/mmdetection/fire-models/latest.pth"
config_fname = "/media/preeth/Data/prajna_files/mmdet_local/mmdetection/fire-models/ssd300_vocdata.py"
score_thr = 0.55

# build the model from a config file and a checkpoint file
model = init_detector(config_fname, checkpoint_file)

image_path = "/media/preeth/Data/prajna_files/mmdet_local/mmdetection/fire-models/input/dataset_file/"
out_dir = "/media/preeth/Data/prajna_files/mmdet_local/mmdetection/fire-models/output/"
image_list = []
images = os.listdir(image_path)
for image in images:    
    img = image_path + image    
    result = inference_detector(model, img)
    outpath = out_dir+image
    print(outpath)
    print(result,"result")
    show_result(img, result, model.CLASSES, score_thr=score_thr, out_file=outpath)
