python
import os
import shutil

class ImageClassifier:
    def __init__(self, path):
        self.path = path
    
    def classify_images(self):
        result = os.listdir(self.path)
        if not os.path.exists('./train'):
            os.mkdir('./train')
        if not os.path.exists('./train/1'):
            os.mkdir('./train/1')
        if not os.path.exists('./test'):
            os.mkdir('./test')
        if not os.path.exists('./test/1'):
            os.mkdir('./test/1')
        num = 0
        for i in result:
            num += 1
            if num % 3 == 0:
                shutil.copyfile(self.path + '/' + i, './train/1' + '/' + i)
            if num % 3 == 1:
                shutil.copyfile(self.path + '/' + i, './train/1' + '/' + i)
            if num % 3 == 2:
                shutil.copyfile(self.path + '/' + i, './test/1' + '/' + i)
        print('分类完成')

path = './Temporary_folder'
classifier = ImageClassifier(path)
classifier.classify_images()
