python

class ImageBox(QWidget):
    def __init__(self):
        super(ImageBox, self).__init__()
        self.img = None
        self.scaled_img = None
        self.point = QPoint(0, 0)
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.scale = 1

    def init_ui(self):
        self.setWindowTitle("ImageBox")

    def set_image(self, img_path):
        """
        open image file
        :param img_path: image file path
        :return:
        """
        # img = QImageReader(img_path)
        # img.setScaledSize(QSize(self.size().width(), self.size().height()))
        # img = img.read()
        self.img = QPixmap(img_path)
        self.scaled_img = self.img

    def paintEvent(self, e):
        """
        receive paint events
        :param e: QPaintEvent
        :return:
        """
        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(self.point, self.scaled_img)
            painter.end()

    def mouseMoveEvent(self, e):
        """
        mouse move events for the widget
        :param e: QMouseEvent
        :return:
        """
        if self.left_click:
            size = self.size().width()
            self.end_pos = e.pos() - self.start_pos
            self.point = self.point + self.end_pos
            self.start_pos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        """
        mouse press events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()

    def mouseReleaseEvent(self, e):
        """
        mouse release events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.LeftButton:
            self.left_click = False

class MainDemo(QWidget):
    def __init__(self):
        super(MainDemo, self).__init__()

        self.setWindowTitle("Image Viewer")
        self.setFixedSize(1000, 600)

        self.open_file = QPushButton("Open Image")
        self.open_file.setToolTip("Open the image to view.")
        self.open_file.clicked.connect(self.open_image)
        self.open_file.setFixedSize(150, 30)

        self.zoom_in = QPushButton("")
        self.zoom_in.clicked.connect(self.large_click)
        self.zoom_in.setFixedSize(30, 30)
        in_icon = QIcon("icons/zoomIn.png")
        self.zoom_in.setIcon(in_icon)
        self.zoom_in.setIconSize(QSize(30, 30))

        self.zoom_out = QPushButton("")
        self.zoom_out.clicked.connect(self.small_click)
        self.zoom_out.setFixedSize(30, 30)
        out_icon = QIcon("icons/zoomOut.png")
        self.zoom_out.setIcon(out_icon)
        self.zoom_out.setIconSize(QSize(30, 30))

        w = QWidget(self)
        layout = QHBoxLayout()
        layout.addWidget(self.open_file)
        layout.addWidget(self.zoom_in)
        layout.addWidget(self.zoom_out)
        layout.setAlignment(Qt.AlignLeft)
        w.setLayout(layout)
        w.setFixedSize(300, 50)

        self.box = ImageBox()
        self.box.setFixedSize(500,500)

        layout = QVBoxLayout()
        layout.addWidget(w)
        layout.addWidget(self.box)
        self.setLayout(layout)

    def open_image(self):
        """
        select image file and open it
        :return:
        """
        # img_name, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.jpg;;*.png")
        img_name, _ = QFileDialog.getOpenFileName(self, "Open Image File","","All Files(*);;*.jpg;;*.png;;*.jpeg")
        self.box.set_image(img_name)

    def large_click(self):
        """
        used to enlarge image
        :return:
        """
        if self.box.scale < 2:
            self.box.scale += 0.1
            self.box.adjustSize()
            self.update()

    def small_click(self):
        """
        used to reduce image
        :return:
        """
        if self.box.scale > 0.3:
            self.box.scale -= 0.2
            self.box.adjustSize()
            self.update()

