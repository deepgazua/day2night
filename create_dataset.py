import numpy as np
import os
import cv2


def create_dataset(dataset_name):
    dataset = [[], []]
    npz_path = os.path.join("dataset", "%s.npz" % dataset_name)

    for i, name in enumerate(["train", "test"]):
        img_dir = os.path.join("dataset", dataset_name, name)
        dataset[i] = np\
            .array([cv2.imread(os.path.join(img_dir, img)).flatten() for img in os.listdir(img_dir)])\
            .astype('float32') / 255

    np.savez(npz_path, train=dataset[0], test=dataset[1])


for d in ['monet', 'photo']:
    create_dataset(d)
