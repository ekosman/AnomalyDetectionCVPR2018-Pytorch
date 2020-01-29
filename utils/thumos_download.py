import os
import json
import threading
from tqdm import tqdm
import numpy as np


# specify download directory
directory = 'tmp'
videoCounter = 0

num_videos = 1010
keys = list(range(1, num_videos + 1))
bar = tqdm(total=num_videos)
f = open('errors.txt', 'w')


def download(keys):
    for key in keys:
        # take video
        # take url of video
        url = f"https://www.crcv.ucf.edu/THUMOS14/Validation_set/videos/video_validation_{key:07}.mp4"
        cmd = f"wget {url} --quiet"
        try:
            os.system(cmd)
        except:
            f.write(url + '\n')
            f.flush()

        bar.update(1)


def main():
    num_threads = 20
    keys_per_thread = int(np.ceil(num_videos / num_threads))
    threads = []
    for i in range(num_threads):
        chunk = keys[i * keys_per_thread: (i + 1) * keys_per_thread]
        t = threading.Thread(target=download, args=(chunk,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print('Download done!')


if __name__ == '__main__':
    main()
    f.close()
