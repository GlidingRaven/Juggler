import numpy as np
from skimage.draw import circle_perimeter_aa
import csv
import random


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros(size, dtype=np.float64)#uint8???

    # Circle
    row = np.random.randint(size[1])
    col = np.random.randint(size[0])
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += random.uniform(0.01, 1) * noise * np.random.rand(*img.shape)
    return (row, col, rad), img

path = 'E:/'

def train_set():
    number_of_images = 200000
    level_of_noise = 3.5
    img_size = (640, 480)
    max_rad = 130
    with open(path+"train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(12516, number_of_images):
            params, img = noisy_circle(img_size, max_rad, level_of_noise)
            np.save(path+"datasets/train/" + str(i) + ".npy", img)
            write(outFile, [path+"datasets/train/" + str(i) + ".npy", params[0], params[1], params[2]])
            if i % 1000 == 0:
                print(i, ' was made')
        print('done')


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    train_set()
