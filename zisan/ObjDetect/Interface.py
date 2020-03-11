#################################################################
### @author：郑晋图 JintuZheng
### Email: jintuzheng@outlook.com
### Remarks: Please respect my labor achievements, and contact me to modify the source code
### Date:2020-03-11
################################################################

import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .test import test  # Import test.py to get mAP after each epoch
from .models import *
from .utils.datasets import *
from .utils.utils import *

from glob import glob
import os
import random
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
import cv2

import warnings
warnings.filterwarnings('ignore')


class ObjDetect_Preprocess(object):
    def __init__(self,classnames=['REC'],currentpath='',datapath='data',xmlpath='Annotations',setspath='ImageSets',labelpath='labels',imgspath='images'):        
        self.sets = ['train', 'test', 'val']
        self.datapath=currentpath+'/'+datapath
        self.classes = classnames
        self.xmlpath=self.datapath+'/'+xmlpath
        self.setspath=self.datapath+'/'+setspath
        self.labelspath=self.datapath+'/'+labelpath
        self.imgspath=self.datapath+'/'+imgspath
        self.makeDataLog()
        self.makeTxt()
        self.voc_label_make()
    
    def makeDataLog(self):
        names=open(self.datapath+'/classes.names','w')

        for i,item in enumerate(self.classes):
            names.write(item)
            if i != (len(self.classes)-1):
                names.write('\n')

        #shapefile=open(self.datapath+'/pics.shapes','w')

        globalconfig = open(self.datapath+'/global_config.data','w')
        #ids=['classes=','train=','names=','backup=','eval=']
        globalconfig.write('classes='+str(len(self.classes))+'\n')
        globalconfig.write('train='+str(self.datapath+'/train.txt\n'))
        globalconfig.write('valid='+str(self.datapath+'/test.txt\n'))
        globalconfig.write('names='+self.datapath+'/classes.names\n')
        globalconfig.write('eval=MYSET\n')
        globalconfig.write('backup=backup')

    def makeTxt(self):
        trainval_percent = 0.1
        train_percent = 0.9
        xmlfilepath = self.xmlpath
        txtsavepath = self.setspath
        total_xml = os.listdir(xmlfilepath)

        num = len(total_xml)
        list = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        ftrainval = open(txtsavepath+'/trainval.txt', 'w')
        ftest = open(txtsavepath+'/test.txt', 'w')
        ftrain = open(txtsavepath+'/train.txt', 'w')
        fval = open(txtsavepath+'/val.txt', 'w')

        for i in list:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftest.write(name)
                else:
                    fval.write(name)
            else:
                ftrain.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()


    def convert(self,size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)


    def convert_annotation(self,image_id):
        in_file = open(self.xmlpath+'/%s.xml' % (image_id))
        out_file = open(self.labelspath+'/%s.txt' % (image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text #2020_2_6
            cls = obj.find('name').text
            #if cls not in self.classes or int(difficult) == 1:
            if cls not in self.classes:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    def voc_label_make(self):
        wd = getcwd()
        #print(wd)
        for image_set in self.sets:
            if not os.path.exists(self.labelspath+'/'):
                os.makedirs(self.labelspath+'/')
            image_ids = open(self.setspath+'/%s.txt' % (image_set)).read().strip().split()
            list_file = open(self.datapath+'/%s.txt' % (image_set), 'w')
            for image_id in image_ids:
                list_file.write(self.imgspath+'/%s.jpg\n' % (image_id))
                self.convert_annotation(image_id)
            list_file.close()
    
    def clear_data(self,is_all=False):
        if is_all==True:
            shutil.rmtree(self.datapath)
            os.mkdir(self.datapath)
            os.mkdir(self.datapath+'\Annotations')
            os.mkdir(self.datapath+'\images')
            os.mkdir(self.datapath+'\ImageSets')
            os.mkdir(self.datapath+'\labels')
        else:
            for i in glob(self.datapath+'/*.txt'):
                os.remove(i)
            for i in glob(self.datapath+'/*.names'):
                os.remove(i)
            for i in glob(self.datapath+'/*.data'):
                os.remove(i)
            for i in glob(self.labelspath+'/*.txt'):
                os.remove(i)
            for i in glob(self.setspath+'/*.txt'):
                os.remove(i)
            os.remove(self.datapath+'/pics.shapes')





class ObjDetect_train(object):
    def __init__(self,currentpath):
        # self.hyperparameters: train.py --evolve --epochs 2 --img-size 320, Metrics: 0.204      0.302      0.175      0.234 (square smart)
        self.hyp = {'xy': 0.1,  # xy loss gain  (giou is about 0.02)
            'wh': 0.1,  # wh loss gain
            'cls': 0.04,  # cls loss gain
            'conf': 4.5,  # conf loss gain
            'iou_t': 0.5,  # iou target-anchor training threshold
            'lr0': 0.001,  # initial learning rate
            'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
            'momentum': 0.90,  # SGD momentum
            'weight_decay': 0.0005}  # optimizer weight decay

        #default:
        self.currentpath=currentpath
        self.global_epochs=68
        self.global_batch_size=8
        self.global_accumulate=8
        self.global_cfg=currentpath+'/cfgs/yolov3-spp.cfg'
        self.global_data_cfg=currentpath+'/data/global_config.data'
        self.global_multi_scale=False
        self.global_img_size=416
        self.global_resume=False
        self.global_transfer=False
        self.global_num_workers=4
        self.global_backend='ncll'
        self.global_nosave=False
        self.global_notest=False
        self.global_evolve=False
        self.global_var=0
        self.Outputpath=self.currentpath+'/weights/'
        self.weightspath=self.currentpath+'/weights/'
        self.device= torch_utils.select_device()

    def rawtrain(self,
            cfg,
            data_cfg,
            img_size=416,
            resume=False,
            epochs=100,  # 500200 batches at bs 4, 117263 images = 68 epochs
            batch_size=16,
            accumulate=4,  # effective bs = 64 = batch_size * accumulate
            freeze_backbone=False,
            transfer=False  # Transfer learning (train only YOLO layers)
        ):
        self.rect=False
        if isinstance(img_size,int):
            self.rect=False
        else:
            self.rect=True
            
        init_seeds()
        weights = self.weightspath
        output=self.Outputpath

        latest = output + 'latest.pth'
        best = output + 'best.pth'

        
        torch.backends.cudnn.benchmark = True  # possibly unsuitable for multiscale
        img_size_test = img_size  # image size for testing

        if self.global_multi_scale:
            img_size_min = round(img_size / 32 / 1.5)
            img_size_max = round(img_size / 32 * 1.5)
            img_size = img_size_max * 32  # initiate with maximum multi_scale size

        # Configure run
        data_dict = parse_data_cfg(data_cfg)
        train_path = data_dict['train']
        nc = int(data_dict['classes'])  # number of classes

        # Initialize model
        model = Darknet(cfg).to(self.device)

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=self.hyp['lr0'], momentum=self.hyp['momentum'], weight_decay=self.hyp['weight_decay'])

        cutoff = -1  # backbone reaches to cutoff layer
        start_epoch = 0
        best_loss = float('inf')
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
        
        ##################################################################################################
        if resume:  # Load previously saved model
            if transfer:  # Transfer learning
                chkpt = torch.load(weights + 'yolov3-spp.pth', map_location=self.device)
                model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                    strict=False)
                for p in model.parameters():
                    p.requires_grad = True if p.shape[0] == nf else False

            else:  # resume from latest.pth
                chkpt = torch.load(latest, map_location=self.device)  # load checkpoint
                model.load_state_dict(chkpt['model'])

            start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_loss = chkpt['best_loss']
            del chkpt

        else:  # Initialize model with backbone (optional)
            if '-tiny.cfg' in cfg:
                cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.weights')
            elif '-spp.cfg' in cfg:
                cutoff = load_darknet_weights(model, weights + 'yolov3-spp.weights')
            elif 'v3.cfg' in cfg:
                cutoff = load_darknet_weights(model, weights + 'yolov3.weights')


        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(self.global_epochs * x) for x in (0.8, 0.9)], gamma=0.1)
        scheduler.last_epoch = start_epoch - 1

        
        # Dataset
        dataset = LoadImagesAndLabels(train_path,
                                    img_size,
                                    batch_size,
                                    augment=True,
                                    rect=self.rect)


        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=self.global_num_workers,
                                shuffle=True,  # disable rectangular training if True
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

        mixed_precision = False
        if mixed_precision:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        # Remove old results
        for f in glob('*_batch*.jpg') + glob('results.txt'):
            os.remove(f)

        # Start training
        model.hyp = self.hyp  # attach self.hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self.device)  # attach class weights
        model_info(model)
        nb = len(dataloader)
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
        n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
        t, t0 = time.time(), time.time()
        Endepoch=0
        for epoch in range(start_epoch, epochs):
            model.train()
            print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'targets', 'time'))

            # Update scheduler
            scheduler.step()

            # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
            if freeze_backbone and epoch < 2:
                for name, p in model.named_parameters():
                    if int(name.split('.')[1]) < cutoff:  # if layer < 75
                        p.requires_grad = False if epoch == 0 else True

            # # Update image weights (optional)
            # w = model.class_weights.cpu().numpy() * (1 - maps)  # class weights
            # image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            # dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # random weighted index

            mloss = torch.zeros(5).to(self.device)  # mean losses
            for i, (imgs, targets, _, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                # Multi-Scale training
                if self.global_multi_scale:
                    if (i + 1 + nb * epoch) % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                        img_size = random.choice(range(img_size_min, img_size_max + 1)) * 32
                        print('img_size = %g' % img_size)
                    scale_factor = img_size / max(imgs.shape[-2:])
                    imgs = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)

                # Plot images with bounding boxes
                if i == (len(dataloader)-1):
                    plot_images(imgs=imgs, targets=targets, fname='{}train_batch{}.jpg'.format(epoch,i))

                # SGD burn-in
                if epoch == 0 and i <= n_burnin:
                    lr = self.hyp['lr0'] * (i / n_burnin) ** 4
                    for x in optimizer.param_groups:
                        x['lr'] = lr

                # Run model
                pred = model(imgs)

                # Compute loss
                loss, loss_items = compute_loss(pred, targets, model)
                if torch.isnan(loss):
                    print('WARNING: nan loss detected, ending training')
                    return results

                # Compute gradient
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Accumulate gradient for x batches before optimizing
                if (i + 1) % accumulate == 0 or (i + 1) == nb:
                    optimizer.step()
                    optimizer.zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                s = ('%8s%12s' + '%10.3g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1),
                    '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
                t = time.time()
                print(s)

            # Calculate mAP (always test final epoch, skip first 5 if self.global_nosave)
            if not (self.global_notest or (self.global_nosave and epoch < 10)) or epoch == epochs - 1:
                with torch.no_grad():
                    results, maps = test(cfg, data_cfg, batch_size=batch_size, img_size=img_size_test, model=model,
                                            conf_thres=0.1)

            # Write epoch results
            with open('results.txt', 'a') as file:
                file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

            # Update best loss
            test_loss = results[4]
            if test_loss < best_loss:
                best_loss = test_loss

            # Save training results
            save = (not self.global_nosave) or (epoch == epochs - 1)
            if save:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                        'best_loss': best_loss,
                        'model': model.module.state_dict() if type(
                            model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                        'optimizer': optimizer.state_dict()}

                # Save latest checkpoint
                torch.save(chkpt, latest)

                # Save best checkpoint
                if best_loss == test_loss:
                    torch.save(chkpt, best)

                # Save backup every 10 epochs (optional)
                if epoch > 0 and epoch % 10 == 0:
                    torch.save(chkpt, weights + 'backup%g.pth' % epoch)

                # Delete checkpoint
                del chkpt

            Endepoch=epoch

        dt = (time.time() - t0) / 3600
        print('%g epochs completed in %.3f hours.' % (Endepoch - start_epoch + 1, dt))
        return results


    def print_mutation(self,hyp,results):
        # Write mutation results
        a = '%11s' * len(hyp) % tuple(hyp.keys())  # self.hyperparam keys
        b = '%11.4g' * len(hyp) % tuple(hyp.values())  # self.hyperparam values
        c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
        print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
        with open('evolve.txt', 'a') as f:
            f.write(c + b + '\n')


    def Run(self,
                epochs=68,
                batch_size=8,
                accumulate=8,
                cfg='yolov3-spp.cfg',
                multi_scale=False,
                img_size=416,
                resume=False,
                transfer=False,
                num_workers=4,
                backend='ncll',
                nosave=False,
                notest=False,
                evolve=False,
                var=0
                ):
        self.global_epochs=epochs
        self.global_batch_size=batch_size
        self.global_accumulate=accumulate
        self.global_cfg=self.currentpath+'/cfgs/'+cfg
        self.global_multi_scale=multi_scale
        self.global_img_size=img_size
        self.global_resume=resume
        self.global_transfer=transfer
        self.global_num_workers=num_workers
        self.global_backend=backend
        self.global_nosave=nosave
        self.global_notest=notest
        self.global_evolve=evolve
        self.global_var=var


        if self.global_evolve:
            self.global_notest = True  # save time by only testing final epoch
            self.global_nosave = True  # do not save checkpoints

        # Train
        results = self.rawtrain(
            self.global_cfg,
            self.global_data_cfg,
            img_size=self.global_img_size,
            resume=self.global_resume or self.global_transfer,
            transfer=self.global_transfer,
            epochs=self.global_epochs,
            batch_size=self.global_batch_size,
            accumulate=self.global_accumulate
        )

        # Evolve self.hyperparameters (optional)
        if self.global_evolve:
            best_fitness = results[2]  # use mAP for fitness

            # Write mutation results
            self.print_mutation(self.hyp, results)

            gen = 1000  # generations to evolve
            for _ in range(gen):

                # Mutate self.hyperparameters
                old_hyp = self.hyp.copy()
                init_seeds(seed=int(time.time()))
                s = [.3, .3, .3, .3, .3, .3, .3, .03, .3]  # xy, wh, cls, conf, iou_t, lr0, lrf, momentum, weight_decay
                for i, k in enumerate(self.hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                    self.hyp[k] = self.hyp[k] * float(x)  # vary by about 30% 1sigma

                # Clip to limits
                keys = ['lr0', 'iou_t', 'momentum', 'weight_decay']
                limits = [(1e-4, 1e-2), (0, 0.90), (0.70, 0.99), (0, 0.01)]
                for k, v in zip(keys, limits):
                    self.hyp[k] = np.clip(self.hyp[k], v[0], v[1])

                # Determine mutation fitness
                results = train(
                    self.global_cfg,
                    self.global_data_cfg,
                    img_size=self.global_img_size,
                    resume=self.global_resume or self.global_transfer,
                    transfer=self.global_transfer,
                    epochs=self.global_epochs,
                    batch_size=self.global_batch_size,
                    accumulate=self.global_accumulate
                )
                mutation_fitness = results[2]

                # Write mutation results
                self.print_mutation(self.hyp, results)

                # Update self.hyperparameters if fitness improved
                if mutation_fitness > best_fitness:
                    # Fitness improved!
                    print('Fitness improved!')
                    best_fitness = mutation_fitness
                else:
                    self.hyp = old_hyp.copy()  # reset self.hyp to

class ObjDetect_detect(object):
    def __init__(self,cfg,currentpath='',img_size=416):
                
        self.currentpath=currentpath
        self.cfg=currentpath+'/cfgs/'+cfg
        self.data_cfg=currentpath+'/data/global_config.data'

        self.device= torch_utils.select_device()
        # Initialize model
        self.model = Darknet(self.cfg, img_size)      
        # Load weights
        self.weightspath=currentpath+'/weights/best.pth'
        '''
        if '-tiny.cfg' in cfg:
              self.weightspath=currentpath+'/weights/' + 'yolov3-tiny.weights'
        elif '-spp.cfg' in cfg:
              self.weightspath=currentpath+'/weights/' + 'yolov3-spp.weights'
        elif 'v3.cfg' in cfg:
              self.weightspath=currentpath+'/weights/'+ 'yolov3.weights'
        '''
        

        self.model.load_state_dict(torch.load(self.weightspath, map_location=self.device)['model'])
        # Fuse Conv2d + BatchNorm2d layers
        self.model.fuse()
        # Eval mode
        self.model.to(self.device).eval()

    
    

    def detect_from_RGBimg(self,
        img,
        img_size_hw=(416,416), #hw
        conf_thres=0.5,
        nms_thres=0.5,
        is_showPreview=False,
        log_print=False):

        #img=img.resize((img_size_hw[1],img_size_hw[0]),Image.ANTIALIAS) #resize
        img=np.array(img,dtype=np.uint8) #change numpy array
        model=self.model 
        classes = load_classes(parse_data_cfg(self.data_cfg)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        im0=img.copy() # copy raw RGB
        #print(img.shape)
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 

        img = torch.from_numpy(img).unsqueeze(0).to(self.device) # add dim and change to float
        #print(img.shape)
        img=img.permute(0,3,1,2) # dims change positions
        #print(img.shape)

        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        #t = time.time()
        result_boxes=[]
        if det is not None and len(det) > 0:
                # Rescale boxes from 416 to true image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Print results to screen
                #print('%gx%g ' % img.shape[2:], end='')  # print image size
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    if log_print:
                        print('%g %s\n' % (n, classes[int(c)]))
                    sp='%g'%(n)
             
                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in det:
                
                    # Add bbox to the image
                    label = '%s' % (classes[int(cls)])
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    if log_print:
                        print("find:x[%d %d],y[%d %d]"%(c1[0],c2[0],c1[1],c2[1]))
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                    box={'class':classes[int(cls)],'x0':c1[0],'x1':c2[0],'y0':c1[1],'y1':c2[1]}
                    result_boxes.append(box)

            

        if log_print:
            print('Done. (%.3fs)' % (time.time() - t))


        if is_showPreview:  # Save image with detections
            im0[:,:,(0,1,2)]=im0[:,:,(2,1,0)]#通道转换
            cv2.imshow('PreviewDetect',im0)
            cv2.waitKey(0)
        
        return result_boxes



