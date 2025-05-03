import argparse
import time
from pathlib import Path
import json
import datetime
from datetime import datetime
from pyzbar.pyzbar import decode
import numpy as np
import os


import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

file = open("lock_data.json", "w")


def detect(save_img=False):
    global kill_count
    kill_count = 0
    global is_detected
    is_detected = False
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        #görüntü boyutları

        #başlangıçta merkez noktasını görüntünün orta noktası olarak belirleme
        


        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            height, width = im0.shape[:2]

            kx0 = int(width*0.25)
            ky0 = int(height*0.1)
            kx1 = int(width*0.75)
            ky1 = int(height*0.90)

            kxm = int((kx0+kx1)/2)
            kym = int((ky0+ky1)/2)

            center = (int(width/2), int(height/2))
            cv2.rectangle(im0, (int(width*0.25),int(height*0.1)),(int(width*0.75),int(height*0.90)), (255,255,255), 5,cv2.LINE_AA)
            cv2.rectangle(im0, (int(width*0.25),int(height*0.1)),(int(width*0.75),int(height*0.90)), (117,67,19), 3,cv2.LINE_AA)
            #cv2.circle(im0, (int(width*0.5),int(height*0.5)), radius=2, color=(255, 255, 255), thickness=-5)
            cv2.line(im0, (int(width/2),int(height/2)-15), (int(width/2),int(height/2)-30), (255,255,255), 3,cv2.LINE_AA) #dikey üst
            cv2.line(im0, (int(width/2),int(height/2)+15), (int(width/2),int(height/2)+30), (255,255,255), 3,cv2.LINE_AA) #dikey alt
            cv2.line(im0, (int(width/2)-15,int(height/2)), (int(width/2)-30,int(height/2)), (255,255,255), 3,cv2.LINE_AA) #yatay sol
            cv2.line(im0, (int(width/2)+15,int(height/2)), (int(width/2)+30,int(height/2)), (255,255,255), 3,cv2.LINE_AA) #yatay sağ
            cv2.line(im0, (int(width/2),int(height/2)-15), (int(width/2),int(height/2)-30), (117,67,19), 2,cv2.LINE_AA) #dikey üst
            cv2.line(im0, (int(width/2),int(height/2)+15), (int(width/2),int(height/2)+30), (117,67,19), 2,cv2.LINE_AA) #dikey alt
            cv2.line(im0, (int(width/2)-15,int(height/2)), (int(width/2)-30,int(height/2)), (117,67,19), 2,cv2.LINE_AA) #yatay sol
            cv2.line(im0, (int(width/2)+15,int(height/2)), (int(width/2)+30,int(height/2)), (117,67,19), 2,cv2.LINE_AA) #yatay sağ
            #cv2.putText(im0, str(datetime.datetime(now)), (20, 30), cv2.FONT_HERSHEY_PLAIN, 2)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H.%M.%S:%f")[:-3]
            dt_string_4 = now.strftime("%H:%M:%S:%f")[:-3]
            #--------------------------
            #saat2 = dt_string_4[0:2:]
            #dakika2 = dt_string_4[3:5:]
            #saniye2 = dt_string_4[6:8:]
            #milisaniye2 = dt_string_4[9:]
            otonom_kilitlenme = 0
            #--------------------------
            with open('C:/Users/savana/Desktop/yolov7_uav/yolov7-uav/sunucu_saati.json', 'r') as f_saat_sv:
                saat_veri = json.load(f_saat_sv)
            
            saat_sv = saat_veri["saat"]
            dakika_sv = saat_veri["dakika"]
            saniye_sv = saat_veri["saniye"]
            milisaniye_sv = saat_veri["milisaniye"]
            
            cv2.putText(
                  img = im0,
                  text = dt_string,
                  org = (20, 30),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (0,0,0),
                  thickness = 3,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "LOCK TIMER: ",
                  org = (20, 60),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (0,0,0),
                  thickness = 3,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "OBJECT WIDTH: %",
                  org = (20, 90),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (0,0,0),
                  thickness = 3,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "OBJECT HEIGHT: %",
                  org = (20, 120),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (0,0,0),
                  thickness = 3,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "DOWN COUNT: ".format(kill_count),
                  org = (20, 150),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (0,0,0),
                  thickness = 3,lineType = cv2.LINE_AA
                )
            #TEXT ^____ OUTLINE
            cv2.putText(
                  img = im0,
                  text = dt_string,
                  org = (20, 30),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (255, 255, 255),
                  thickness = 1,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "LOCK TIMER: ",
                  org = (20, 60),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (255, 255, 255),
                  thickness = 1,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "OBJECT WIDTH: %",
                  org = (20, 90),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (255, 255, 255),
                  thickness = 1,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "OBJECT HEIGHT: %",
                  org = (20, 120),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (255, 255, 255),
                  thickness = 1,lineType = cv2.LINE_AA
                )
            cv2.putText(
                  img = im0,
                  text = "DOWN COUNT: ".format(kill_count),
                  org = (20, 150),
                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                  fontScale = 0.9,
                  color = (255, 255, 255),
                  thickness = 1,lineType = cv2.LINE_AA
                )
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh




            cooldown = 30
            last_detection_time = 0
            qr_data = []
            
            for barcode in decode(im0):
                current_time = time.time()
                if current_time - last_detection_time > cooldown:
                    mydata = barcode.data.decode('utf-8')
                    detection_time = time.localtime(current_time)
                    detection_time_dict = {
                        'saat': detection_time.tm_hour,
                        'dakika': detection_time.tm_min,
                        'saniye': detection_time.tm_sec,
                        'milisaniye': int((current_time - int(current_time)) * 1000)
                    }
                    qr_data.append({
                        'kamikazeBaslangicZamani': detection_time_dict,
                        'kamikazeBitisZamani': detection_time_dict,
                        'qrMetni': mydata
                    })
                    print(mydata)
                    pts = np.array([barcode.polygon], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(im0, [pts], True, (255, 0, 255), 3)
                    pts2 = barcode.rect
                    cv2.putText(im0, mydata, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                    last_detection_time = current_time
            
            desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            file_path = os.path.join(desktop_path, 'qr_data.json')
            with open(file_path, 'w') as f:
                json.dump(qr_data, f, indent=4)

               
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            

                # Print results
            for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
            min_distance = float("inf")
            for *xyxy, conf, cls in reversed(det):
                    # EDITING HERE !!!!!
                x1 = xyxy[0].item() # X1 coordinate of bounding box
                y1 = xyxy[1].item() # Y1 coordinate of bounding box
                x2 = xyxy[2].item() # X2 coordinate of bounding box
                y2 = xyxy[3].item() # Y2 coordinate of bounding box
                bbox_center = [(x1+x2)/2, (y1+y2)/2] # center of bounding box


                distance = ((center[0] - bbox_center[0])**2 + (center[1] - bbox_center[1])**2)**0.5
                cv2.circle(im0, (int((x1+x2)/2), int((y1+y2)/2)), radius=2, color=(255, 255, 255), thickness=-5)
                cv2.line(im0, (int(width/2),int(height/2)), (int((x1+x2)/2), int((y1+y2)/2)), (255,255,255), 4,cv2.LINE_AA)
                cv2.line(im0, (int(width/2),int(height/2)), (int((x1+x2)/2), int((y1+y2)/2)), (117,67,19), 2,cv2.LINE_AA)
                    

                if distance < min_distance:
                    min_distance = distance
                    #closest_bbox = bbox
                    
                    #seçilen bbox'un orta noktasını yeni merkez nokta olarak ayarlama
                    #closest_bbox_center = ((closest_bbox[0] + closest_bbox[2])/2, (closest_bbox[1] + closest_bbox[3])/2)
                    center = bbox_center #closest_bbox_center
                    xyxy_final = xyxy
                    conf_final = conf

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                
                lock_color = (0,0,255)        
                if save_txt:  # Write to file
                    if ( (x1 > kx0 and x1 < kx1) and (y1 > ky0 and y1 < ky1) ) and ( (x2 > kx0 and x2 < kx1) and (y2 > ky0 and y2 < ky1) ):
                        #lock_color = (0,255,0)
                        xywh_final = (xyxy2xywh(torch.tensor(xyxy_final).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh_final, conf_final) if opt.save_conf else (cls, *xywh_final)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


                bbox_height = y2-y1
                bbox_width = x2-x1
                bbox_height_percantage = ((bbox_height/height)*100)
                bbox_width_percantage = ((bbox_width/width)*100)
                #bbox_area = bbox_height*bbox_width
                #screen_area = height*width
                #object_size = float(((bbox_area/screen_area))*100)
                
                if ( (x1 > kx0 and x1 < kx1) and (y1 > ky0 and y1 < ky1) ) and ( (x2 > kx0 and x2 < kx1) and (y2 > ky0 and y2 < ky1) and ((bbox_height_percantage) > 6 or (bbox_width_percantage) > 6)):
                        #lock_color = (0,255,0)
                #if lock_color == (0,255,0):
                        cv2.putText(
                              img = im0,
                              text = "TARGET LOCKED SUCCESSFULLY",#.format(bbox_height_percantage),
                              org = (kxm-280, 680),
                              fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              fontScale = 1.5,
                              color = (0, 125, 0),
                              thickness = 3,lineType = cv2.LINE_AA
                            )
                        cv2.putText(
                              img = im0,
                              text = "TARGET LOCKED SUCCESSFULLY",#.format(bbox_height_percantage),
                              org = (kxm-280, 680),
                              fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              fontScale = 1.5,
                              color = (255, 255, 255),
                              thickness = 1,lineType = cv2.LINE_AA
                            )
                        if not is_detected:
                            is_detected = True
                            is_sent = False
                            t0 = time.time()
                                                       
                            with open('C:/Users/savana/Desktop/yolov7_uav/yolov7-uav/sunucu_saati.json' ,'r') as f_saat_sv2:
                                saat_veri2 = json.load(f_saat_sv2)
            
                            saat_sv2 = saat_veri2["saat"]
                            dakika_sv2 = saat_veri2["dakika"]
                            saniye_sv2 = saat_veri2["saniye"]
                            milisaniye_sv2 = saat_veri2["milisaniye"]
                            
                            #tnow_str = str(now.strftime("%H:%M:%S:%f")[:-3])
                            #saat = tnow_str[0:2:]
                            #dakika = tnow_str[3:5:]
                            #saniye = tnow_str[6:8:]
                            #milisaniye = tnow_str[9:]
                        t_diff = time.time() - t0
                        if is_detected and not is_sent:
                            cv2.putText(
                                  img = im0,
                                  text = "LOCK TIMER: {0:.2}".format(t_diff),
                                  org = (20, 60),
                                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                  fontScale = 0.9,
                                  color = (0,0,0),
                                  thickness = 3,lineType = cv2.LINE_AA
                                )
                            cv2.putText(
                                  img = im0,
                                  text = "LOCK TIMER: {0:.2}".format(t_diff),
                                  org = (20, 60),
                                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                  fontScale = 0.9,
                                  color = (255,255,255),
                                  thickness = 1,lineType = cv2.LINE_AA
                                )
                            if t_diff > 4:
                                #OUTPUT DATA 4 SEC

                                data = {
                                        "kilitlenmeBaslangicZamani": {
                                            "saat": saat_sv,
                                            "dakika": dakika_sv,
                                            "saniye": saniye_sv,
                                            "milisaniye": milisaniye_sv
                                        },
                                        "kilitlenmeBitisZamani": {
                                            "saat": saat_sv2,
                                            "dakika": dakika_sv2,
                                            "saniye": saniye_sv2,
                                            "milisaniye": milisaniye_sv2
                                        },
                                        "otonom_kilitlenme": otonom_kilitlenme
                                }

                                uav_kilitlenme = json.dumps(data)
                                
                                with open('kilitlenme_verisi.json', 'w') as outfile:
                                    outfile.write(uav_kilitlenme)

                                #uav_detection_data = json.dump(data, indent=True)
                                #json.load(uav_detection_data)
                                
                                #eski stil
                                #file.write("\nTarget lock started at: {} .".format(tnow_str),)
                                #file.write("\nTarget lock finished at: {} .".format(dt_string_4),)
                                #file.write("\n-----------------------------")
                                kill_count += 1

                                is_sent = True


                        if is_sent:
                            
                            cv2.putText(
                                  img = im0,
                                  text = "TARGET DOWN",#.format(bbox_height_percantage),
                                  org = (kxm-150,int(height/2)+270),
                                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                  fontScale = 1.5,
                                  color = (0, 0, 125),
                                  thickness = 3,lineType = cv2.LINE_AA)
                            cv2.putText(
                                  img = im0,
                                  text = "TARGET DOWN",#.format(bbox_height_percantage),
                                  org = (kxm-150,int(height/2)+270),
                                  fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                  fontScale = 1.5,
                                  color = (255, 255, 255),
                                  thickness = 1,lineType = cv2.LINE_AA
                                )
                            #file.write("target locked 4 second at {}\n".format(dt_string),)
                            #if t_diff > 4:
                            #    file.write("target locked 4 second at {}".format(dt_string),)
                            #    is_sent = True
                        if t_diff > 4:
                            #cv2.putText(
                            #      img = im0,
                            #      text = "TARGET DESTROYED",#.format(bbox_height_percantage),
                            #      org = (110, 160),
                            #      fontFace = cv2.FONT_HERSHEY_PLAIN,
                            #      fontScale = 2.5,
                            #      color = (255, 255, 255),
                            #      thickness = 1
                            #    )
                            is_sent = True
                else:
                     #eklenmE.............E.E.E.E.E.E.E.EE


                     cv2.putText(
                           img = im0,
                           text = "TARGET DETECTED",#.format(bbox_height_percantage),
                           org = (kxm-150, 680),
                           fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                           fontScale = 1.5,
                           color = (255, 255, 255),
                           thickness = 3,lineType = cv2.LINE_AA
                         )

                     

                     cv2.putText(
                           img = im0,
                           text = "TARGET DETECTED",#.format(bbox_height_percantage),
                           org = (kxm-150, 680),
                           fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                           fontScale = 1.5,
                           color = (117,67,19),
                           thickness = 1,lineType = cv2.LINE_AA
                         )
                     is_detected = False
                    #cv2.putText(
                    #      img = im0,
                    #      text = "TARGET SUCCESSFULLY LOCKED",#.format(bbox_height_percantage),
                    #      org = (440, 680),
                    #      fontFace = cv2.FONT_HERSHEY_PLAIN,
                    #      fontScale = 1.5,
                    #      color = (69, 11, 11),
                    #      thickness = 3
                    #    )
                    #cv2.putText(
                    #      img = im0,
                    #      text = "TARGET SUCCESSFULLY LOCKED",#.format(bbox_height_percantage),
                    #      org = (440, 680),
                    #      fontFace = cv2.FONT_HERSHEY_PLAIN,
                    #      fontScale = 1.5,
                    #      color = (255, 255, 255),
                    #      thickness = 1
                    #    )
                        
                
                plot_one_box(xyxy_final, im0, color=lock_color, line_thickness=2) #colors[int(cls)]

                #cv2.putText(
                #      img = im0,
                #      text = "LOCK TIMER: 00.00.00",
                #      org = (20, 60),
                #      fontFace = cv2.FONT_HERSHEY_PLAIN,
                #      fontScale = 0.9,
                #      color = (69,11,11),
                #      thickness = 3
                #    )
                cv2.putText(
                      img = im0,
                      text = "OBJECT WIDTH: %{0:.2f}".format(bbox_width_percantage),
                      org = (20, 90),
                      fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale = 0.9,
                      color = (0,0,0),
                      thickness = 3,lineType = cv2.LINE_AA
                    )
                cv2.putText(
                      img = im0,
                      text = "OBJECT HEIGHT: %{0:.2f}".format(bbox_height_percantage),
                      org = (20, 120),
                      fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale = 0.9,
                      color = (0,0,0),
                      thickness = 3,lineType = cv2.LINE_AA
                    )
                cv2.putText(
                      img = im0,
                      text = "DOWN COUNT: {0}".format(kill_count),
                      org = (20, 150),
                      fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale = 0.9,
                      color = (0,0,0),
                      thickness = 3,lineType = cv2.LINE_AA
                    )
            #TEXT ^____ OUTLINE

                #cv2.putText(
                #      img = im0,
                #      text = "LOCK TIMER: 00.00.00",
                #      org = (20, 60),
                #      fontFace = cv2.FONT_HERSHEY_PLAIN,
                #      fontScale = 0.9,
                #      color = (255, 255, 255),
                #      thickness = 1
                #    )
                cv2.putText(
                      img = im0,
                      text = "OBJECT WIDTH: %{0:.2f}".format(bbox_width_percantage),
                      org = (20, 90),
                      fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale = 0.9,
                      color = (255, 255, 255),
                      thickness = 1,lineType = cv2.LINE_AA
                    )
                cv2.putText(
                      img = im0,
                      text = "OBJECT HEIGHT: %{0:.2f}".format(bbox_height_percantage),
                      org = (20, 120),
                      fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale = 0.9,
                      color = (255, 255, 255),
                      thickness = 1,lineType = cv2.LINE_AA
                    )
                cv2.putText(
                      img = im0,
                      text = "DOWN COUNT: {0}".format(kill_count),
                      org = (20, 150),
                      fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      fontScale = 0.9,
                      color = (255, 255, 255),
                      thickness = 1,lineType = cv2.LINE_AA
                    )

            # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
file.close()
