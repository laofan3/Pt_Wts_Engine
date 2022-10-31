import os


def wts2engine():
    # os.system('yolov5.exe -s detect.wts detect.engine s')
    if not os.path.exists("./detect.wts"):
        print("wts文件不存在")
    else:
        print("开始转换为engine文件，请稍等。。。")
        os.system('yolov5.exe -s detect.wts detect.engine s')


if __name__ == '__main__':
    wts2engine()
