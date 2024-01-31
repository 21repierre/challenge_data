import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

h5f = h5py.File(f'./datasets/data_test_images.h5', 'r')
train_images = h5f['data_test_images'][:]
h5f.close()

expressions = ['happy', 'fear', 'surprise', 'anger', 'disgust', 'sad']
expression_mapping = dict(zip(range(len(expressions)), expressions))

print(expression_mapping)

N = len(train_images)

labels = [
    'anger', 'anger', 'disgust', 'disgust', 'fear', 'fear', 'happy', 'happy', 'sad', 'sad', 'surprise', 'surprise', 'anger', 'anger', 'disgust',
    'disgust', 'anger', 'happy', 'happy', 'sad', 'sad', 'fear', 'fear', 'fear', 'happy', 'surprise', 'disgust', 'surprise', 'anger', 'disgust',
    'disgust', 'anger', 'surprise', 'surprise', 'anger', 'disgust', 'happy', 'happy', 'happy', 'anger', 'fear', 'fear', 'sad', 'sad', 'surprise',
    'surprise', 'anger', 'anger', 'disgust', 'fear', 'happy', 'surprise', 'surprise', 'anger', 'disgust', 'disgust', 'surprise', 'happy', 'happy', 'sad',
    'sad', 'surprise', 'surprise', 'anger', 'disgust', 'fear', 'happy', 'anger', 'surprise', 'fear', 'disgust', 'disgust', 'sad', 'surprise', 'anger',
    'anger', 'surprise', 'fear', 'happy', 'anger', 'disgust', 'happy', 'sad', 'sad', 'surprise', 'surprise', 'anger', 'disgust', 'fear', 'happy',
    'sad', 'sad', 'surprise', 'anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'anger', 'disgust', 'disgust', 'fear', 'fear', 'happy',
    'anger', 'anger', 'anger', 'fear', 'anger', 'surprise', 'disgust', 'fear', 'happy', 'sad', 'fear', 'happy', 'happy', 'anger', 'disgust',
    'surprise', 'anger', 'happy', 'anger', 'anger', 'disgust', 'disgust', 'fear', 'fear', 'happy', 'happy', 'sad', 'sad', 'surprise', 'surprise',
    'anger', 'fear', 'anger', 'anger', 'fear', 'surprise', 'disgust', 'happy', 'happy', 'sad', 'anger', 'anger', 'surprise', 'surprise', 'anger',
    'anger', 'disgust', 'disgust', 'fear', 'happy', 'happy', 'sad', 'sad', 'surprise', 'surprise', 'fear', 'anger', 'disgust', 'disgust', 'happy',
    'anger', 'surprise', 'surprise', 'anger', 'anger', 'anger', 'disgust', 'disgust', 'fear', 'surprise', 'happy', 'happy', 'happy', 'sad', 'sad',
    'sad', 'fear', 'fear', 'surprise', 'surprise', 'disgust', 'disgust', 'anger', 'fear', 'fear', 'happy', 'happy', 'happy', 'sad', 'sad',
    'sad', 'surprise', 'fear', 'happy', 'happy', 'happy', 'happy', 'happy', 'disgust', 'happy', 'disgust']

h5f = h5py.File('./datasets/data_test_labels.h5', 'w')
labels2 = np.array([expressions.index(label) for label in labels])
h5f['data_test_labels'] = labels2
h5f.close()

test_labels_guillaume = [
    3, 3, 4, 4, 5, 5, 0, 0, 1, 1,
    2, 2, 3, 3, 4, 4, 5, 0, 0, 1,
    1, 2, 2, 0, 0, 3, 2, 4, 3, 4,
    2, 2, 2, 2, 0, 0, 0, 0, 3, 1,
    1, 5, 5, 2, 2, 3, 3, 4, 1, 0,
    2, 2, 3, 5, 5, 2, 0, 0, 5, 5,
    2, 2, 4, 4, 1, 0, 5, 2, 3, 2,
    4, 5, 2, 5, 5, 2, 2, 0, 3, 5,
    0, 5, 5, 2, 2, 3, 4, 2, 0, 5,
    5, 2, 3, 4, 4, 0, 5, 2, 4, 4,
    4, 1, 1, 0, 5, 5, 3, 2, 4, 2,
    3, 2, 0, 3, 2, 0, 0, 3, 4, 2,
    2, 0, 3, 3, 3, 3, 1, 1, 0, 0,
    3, 5, 2, 2, 5, 5, 5, 5, 5, 2,
    5, 0, 0, 5, 5, 5, 2, 2, 3, 3,
    4, 4, 2, 0, 0, 5, 5, 2, 2, 0,
    3, 5, 1, 0, 2, 2, 2, 3, 3, 3,
    4, 4, 2, 2, 0, 0, 0, 5, 5, 5,
    2, 2, 2, 2, 1, 4, 1, 1, 1, 0,
    0, 0, 5, 5, 5, 2, 2, 0, 0, 0,
    0, 0, 3, 0, 4
]

print(len(test_labels_guillaume), N)
ct = 0
for i in range(len(test_labels_guillaume)):
    if test_labels_guillaume[i] != labels2[i]:
        print(f"[{i}] Guillaume: {expression_mapping[test_labels_guillaume[i]]:<20}Moi: {expression_mapping[labels2[i]]}")
        ct += 1

print(ct, ct / len(test_labels_guillaume))

# 21 lignes / 10 cols

for i in trange(14):
    fig = plt.figure(figsize=(15, 20))
    for j in range(15):
        if 15 * i + j >= N:
            break
        imgs = train_images[15 * i + j]
        for k in range(10):
            # print(i, j, 10 * j + (k + 1))
            ax = fig.add_subplot(15, 10, 10 * j + (k + 1))
            ax.set_axis_off()
            plt.imshow(imgs[k], cmap='gray')
            if len(labels) > 15 * i + j:
                plt.title(f"{15 * i + j}|{labels[15 * i + j]}")
    fig.tight_layout()
    fig.show()

plt.tight_layout()
plt.axis('off')
plt.show()
