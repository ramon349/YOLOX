# encoding: utf-8
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
from torch import nn 


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.warmup_epochs = 1
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.test_size = (416,416)

        self.enable_mixup = False  


        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
    def get_model(self):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data.datasets.ct_voc import CTAnnotationTransform,CTVOCDetection

        return CTVOCDetection("/media/Datacenter_storage/ramon_dataset_curations/kidney_radiomics_yolo/datasets/stu_bbox_dataset.csv",image_set='train')

    def get_eval_dataset(self, **kwargs):
        from yolox.data.datasets.ct_voc import CTAnnotationTransform,CTVOCDetection

        return CTVOCDetection("/media/Datacenter_storage/ramon_dataset_curations/kidney_radiomics_yolo/datasets/stu_bbox_dataset.csv",image_set='test')

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
