from tkinter import *
from tkinter.filedialog import *
from tkinter.ttk import *

from PIL import Image


class Discern(object):

    def __init__(self):
        self.img = None
        self.window = Tk()
        self.window.title("数字检测")
        self.window.geometry("300x270+80+80")
        upload = Button(self.window, text="上传图片", command=self.upImage)
        upload.pack()
        self.window.mainloop()

    def upImage(self):
        imagePath = askopenfilename(initialdir=sys.path[0])
        img = Image.open(imagePath)
        self.img = img.resize((28, 28), Image.ANTIALIAS).convert("L")

if __name__ == "__main__":



    dis = Discern()
