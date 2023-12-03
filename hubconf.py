python
import torch

class YOLOv5:
    def __init__(self, name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
        self.name = name
        self.pretrained = pretrained
        self.channels = channels
        self.classes = classes
        self.autoshape = autoshape
        self.verbose = verbose
        self.device = device
        self.model = self._create_model()

    def _create_model(self):
        from pathlib import Path
        from models.yolo import Model
        from models.experimental import attempt_load
        from utils.general import check_requirements, set_logging
        from utils.downloads import attempt_download
        from utils.torch_utils import select_device

        file = Path(__file__).resolve()
        check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
        set_logging(verbose=self.verbose)

        save_dir = Path('') if str(self.name).endswith('.pt') else file.parent
        path = (save_dir / self.name).with_suffix('.pt')  # checkpoint path
        try:
            device = select_device(('0' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device)

            if self.pretrained and self.channels == 3 and self.classes == 80:
                model = attempt_load(path, map_location=device)  # download/load FP32 model
            else:
                cfg = list((Path(__file__).parent / 'models').rglob(f'{self.name}.yaml'))[0]  # model.yaml path
                model = Model(cfg, self.channels, self.classes)  # create model
                if self.pretrained:
                    ckpt = torch.load(attempt_download(path), map_location=device)  # load
                    msd = model.state_dict()  # model state_dict
                    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                    csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
                    model.load_state_dict(csd, strict=False)  # load
                    if len(ckpt['model'].names) == self.classes:
                        model.names = ckpt['model'].names  # set class names attribute
            if self.autoshape:
                model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
            return model.to(device)

        except Exception as e:
            help_url = 'https://github.com/ultralytics/yolov5/issues/36'
            s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
            raise Exception(s) from e

    def predict(self, imgs):
        return self.model(imgs)


