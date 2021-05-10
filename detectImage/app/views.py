from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.generic import TemplateView


import random
import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from PIL import Image
import io
import base64


def index(request):
    template_name = '../Templates/index.html'
    return render(request, template_name)


def predictImage(request):
    template_name = '../Templates/detectImg.html'

    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()

    filePathName = fs.save(fileObj.name, fileObj)

    filePathName = fs.url(filePathName)
    filePathName = str(filePathName).lstrip('/')

    detectLabel = ''
    detectProbability = ''

    weights = 'best.pt'
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'
    imgsz = 640

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    desc_file = "xray_desc.csv"
    f = open(desc_file, "r", encoding='UTF-8')
    desc = f.readlines()
    f.close()
    dict = {}
    for line in desc:
        dict[line.split('|')[0]] = line.split('|')[1]

    save_img = True
    dataset = LoadImages(filePathName, img_size=imgsz,
                         stride=stride)  # mình đang fix cứng đường dẫn ảnh input
    img = Image.open(filePathName)
    width, height = img.size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    conf_thres = 0.25
    iou_thres = 0.25

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]  # img chỗ này là ảnh đầu vào, là ảnh Test1 của bạn vừa upload lên

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                   agnostic=False)  # còn pred chỗ này là output của yolov5, bao gồm tọa độ của bounding box

        extra = ""
        coor = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            cai = fileObj.name
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        detectLabel = f'{names[int(cls)]}'
                        detectProbability = f'{conf:.2f}'

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],
                                     line_thickness=3)  # dòng này là hàm để vẽ bounding box lên ảnh input của bạn (là ảnh Test 1)

                        x0 = str(xyxy[0]).lstrip('tensor(')
                        x0 = x0[:-2]
                        x0 = float(x0)
                        x1 = str(xyxy[1]).lstrip('tensor(')
                        x1 = x1[:-2]
                        x1 = float(x1)
                        x2 = str(xyxy[2]).lstrip('tensor(')
                        x2 = x2[:-2]
                        x2 = float(x2)
                        x3 = str(xyxy[3]).lstrip('tensor(')
                        x3 = x3[:-2]
                        x3 = float(x3)
                        coor.append(x0)
                        coor.append(x1)
                        coor.append(x2)
                        coor.append(x3)

                        # extra += "<br>- <b>" + str(names[int(cls)]) + "</b> (" + dict[names[int(cls)]] \
                        #          + ") với độ tin cậy <b>{:.2f}% </b>".format(conf)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite("output_yolov5.jpeg", im0)

    im = Image.fromarray(im0.astype("uint8"))
    # im.show()  # uncomment to look at the image
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    img_uri = base64.b64encode(rawBytes.read())

    img_uri_file = "data:image/png;base64," + str(img_uri)[2:-1]

    if detectLabel == 'Aortic enlargement':
        detectLabel = 'Rộng động mạch chủ'
    if detectLabel == 'Atelectasis':
        detectLabel = 'Xẹp phổi'
    if detectLabel == 'Calcification':
        detectLabel = 'Vôi hóa'
    if detectLabel == 'Cardiomegaly':
        detectLabel = 'Tim to'
    if detectLabel == 'Consolidation':
        detectLabel = 'Consolidation'
    if detectLabel == 'ILD':
        detectLabel = 'Bệnh phổi kẽ'
    if detectLabel == 'Infiltration':
        detectLabel = 'Thâm nhiễm phổi'
    if detectLabel == 'Lung Opacity':
        detectLabel = 'Độ mờ của phổi'
    if detectLabel == 'Nodule/Mass':
        detectLabel = 'Nốt phổi'
    if detectLabel == 'Other lesion':
        detectLabel = 'Tổn thương khác'
    if detectLabel == 'Pleural effusion':
        detectLabel = 'Tràn dịch màng phổi'
    if detectLabel == 'Pleural thickening':
        detectLabel = 'Dày màng phổi'
    if detectLabel == 'Pneumothorax':
        detectLabel = 'Tràn khí màng phổi'
    if detectLabel == 'Pulmonary fibrosis':
        detectLabel = 'Xơ phổi'

    print(coor)
    context = {
        'filePathName': filePathName,
        'img_uri': img_uri_file,
        'label': detectLabel,
        'Probability': detectProbability,
        'coor': coor,
        'width': width,
        'height': height,
    }

    return render(request, template_name, context)


class editOutput(TemplateView):
    template_name = '../Templates/output.html'

    def get(self, request, filePath):
        filePathName = request.get_full_path()
        filePathName = filePathName.replace("/editOutput", "")
        content = {
            'filePath': filePathName,
        }
        return render(request, self.template_name, content)
