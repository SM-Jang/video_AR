import cv2
import argparse
import os
import pdb


def getFrame(sec, vidcap, count, data_name, save_path):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, image = vidcap.read()
    
    if hasFrames:
        name = save_path + '/' + data_name[:-4] + '_' + str(count) + '.jpg'
        cv2.imwrite(name, image)     # save frame as JPG file
    print(count)
    return hasFrames


def main(fr, data, data_name, save_path):
    
    # set path
    data_path = os.path.join(os.getcwd(), data)
    data_path = os.path.join(data_path, data_name)
    
    # main
    vidcap = cv2.VideoCapture(data_path)
    sec = 0
    frameRate = fr #//it will capture image in each second
    count = 1
    success = getFrame(sec, vidcap, count, data_name, save_path)
    
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, vidcap, count, data_name, save_path)
    
def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fr', type=float)
    parser.add_argument('--data', type=str)
    parser.add_argument('--save', type=str)
    
    
    return parser.parse_args()
    
    
if __name__ == '__main__':
    
    args = get_arguments()
    
    save_path = os.path.join(os.getcwd(), args.save)
    
    
    # 폴더 만들기
    for data_name in os.listdir(args.data):
        dir_name = os.path.join(args.save, data_name)
        if dir_name[-4:] == '.avi':
            if os.path.isdir(dir_name[:-4]) == False: os.mkdir(dir_name[:-4])
        
        
        save_path = dir_name[:-4]

        
        print(data_name)
        main(args.fr, args.data, data_name, save_path)


