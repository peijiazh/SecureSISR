import torch.utils.data as data
import torch
import h5py
from PIL import Image


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path="./matlab/train.h5"):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index, :, :, :]).float()
        label = torch.from_numpy(self.target[index, :, :, :]).float()
        return image, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    data = DatasetFromHdf5()
    print(len(data))
    image = data[0][0]
    label = data[0][1]
    print(image.numpy()[0].shape)
    # show images
    image = Image.fromarray(image.numpy()[0] * 255)
    label = Image.fromarray(label.numpy()[0] * 255)
    image.show()
    label.show()
