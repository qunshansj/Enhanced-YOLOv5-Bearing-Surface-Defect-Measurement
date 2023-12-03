python
import cv2

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.HSV = self.image.copy()
        self.HSV2 = self.image.copy()
        self.list = []

    def resize_image(self):
        height, width, channels = self.image.shape
        if width > 1500 or width < 600:
            scale = 1200 / width
            print("图片的尺寸由 %dx%d, 调整到 %dx%d" % (width, height, width * scale, height * scale))
            scaled = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
            return scaled, scale

    def get_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            HSV3 = self.HSV2.copy()
            self.HSV = HSV3
            cv2.line(self.HSV, (0, y), (self.HSV.shape[1] - 1, y), (0, 0, 0), 1, 4)
            cv2.line(self.HSV, (x, 0), (x, self.HSV.shape[0] - 1), (0, 0, 0), 1, 4)
            cv2.imshow("imageHSV", self.HSV)
        elif event == cv2.EVENT_LBUTTONDOWN:
            HSV3 = self.HSV2.copy()
            self.HSV = HSV3
            self.list.append([int(x), int(y)])
            print(self.list[-1])

    def process_image(self):
        cv2.imshow("imageHSV", self.HSV)
        cv2.setMouseCallback("imageHSV", self.get_position)
        cv2.waitKey(0)

image_processor = ImageProcessor('./1.jpg')
image_processor.process_image()
