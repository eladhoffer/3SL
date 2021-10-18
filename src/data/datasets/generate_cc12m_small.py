from torchvision.transforms.transforms import CenterCrop
from cc12m import CCSE
from torchvision import transforms
import numpy as np
transform = transforms.Compose([
    transforms.Resize(size=32),
    transforms.CenterCrop(size=32),
])


def save_as_numpy(transform=transform, sample_csv='/home/labuser/Datasets/cc12m/5K.csv', num=None, output_file='cc12m_small_5K'):
    data = CCSE(sample_csv=sample_csv, transform=transform)
    num = len(data) if num is None else num
    N = min(len(data), num)
    D = data[0][1].size(0)

    image_array = np.zeros((N, 32, 32, 3), dtype=np.uint8)
    embedding_array = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        if i % 1000 == 0:
            print('{}/{}'.format(i, N))
        image, embedding = data[i]
        image_array[i] = np.asarray(image)
        embedding_array[i] = embedding.numpy()
    with open(f'{output_file}_images.npy', 'wb') as f:
        np.save(f, image_array)
    with open(f'{output_file}_embeddings.npy', 'wb') as f:
        np.save(f, embedding_array)


save_as_numpy(sample_csv='/home/labuser/Datasets/cc12m/5K.csv',
              num=5000, output_file='cc12m_small_5K')
save_as_numpy(sample_csv='/home/labuser/Datasets/cc12m/5M.csv',
              num=5000000, output_file='cc12m_small_5M')
