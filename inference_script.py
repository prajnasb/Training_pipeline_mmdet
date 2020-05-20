from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, init_detector
import matplotlib.pyplot as plt
import cv2


%matplotlib inline

checkpoint_file = "/content/mmdetection/docker/latest.pth"
config_fname = "/content/mmdetection/configs/pascal_voc/ssd300_voc0712.py"
score_thr = 0.55

# build the model from a config file and a checkpoint file
model = init_detector(config_fname, checkpoint_file)

# test a single image and show the results
img = '/content/data/data/VOC2007/JPEGImages/1-04.jpg'
predictions = inference_detector(model, img)
results = []
for i, label in enumerate(["washing_hands"]):
    prediction = predictions[i]
    for box in prediction:
        bndbox = box[:4]
        bndbox = [int(x_) for x_ in bndbox]
        confidence = float(box[4])
        if confidence < 0.7:
            continue

        result = {"label":label,"score":confidence,"box":{"x1":bndbox[0],"y1":bndbox[1],"x2":bndbox[2],"y2":bndbox[3]}}
        
        print(result)
x1=int(result["box"]["x1"])
y1=int(result["box"]["y1"])
x2=int(result["box"]["x2"])
y2=int(result["box"]["y2"])
font = cv2.FONT_HERSHEY_SIMPLEX 
       
img=cv2.imread("/content/data/data/VOC2007/JPEGImages/1-04.jpg")
img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
img = cv2.putText(img, label, (x1+20,y1+30),font, 1,(0,0,255) , 2, cv2.LINE_AA) 
#image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA) 
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()
