import cv2
import time


def main(file_path):
    cap = cv2.cudacodec.createVideoReader(file_path)
    # i = 1
    start_time = time.time()
    while True:
        ret, frame = cap.nextFrame()
        if ret is False:
            break
    # frame_cuda = frame.download()
    # print(i)
    # i = i + 1
    # cv2.imshow('image', frame_cuda)
    # cv2.waitKey(1)
    end_time = time.time()
    exe_time = end_time - start_time
    print("time:", exe_time)
    cap = None


if __name__ == '__main__':
    file_path = 'video.mp4'
    main(file_path)
