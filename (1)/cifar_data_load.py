import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def GetPhoto(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]; r = np.reshape(r, [32, 32, 1])
    g = pixel[1024:2048]; g = np.reshape(g, [32, 32, 1])
    b = pixel[2048:3072]; b = np.reshape(b, [32, 32, 1])

    photo = np.concatenate([r, g, b], -1)

    return photo


def GetTrainDataByLabel(label):
    batch_label = []
    labels = []
    data = []
    filenames = []
    for i in range(1, 1+5):
        batch_label.append(unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d"%i)[b'batch_label'])
        labels += unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d"%i)[b'labels']
        data.append(unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d"%i)[b'data'])
        filenames += unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d"%i)[b'filenames']

    data = np.concatenate(data, 0)

    label = str.encode(label)
    if label == b'data':
        array = np.ndarray([len(data), 32, 32, 3], dtype=np.int32)
        for i in range(len(data)):
            array[i] = GetPhoto(data[i])
        return array
        pass
    elif label == b'labels':
        return labels
        pass
    elif label == b'batch_label':
        return batch_label
        pass
    elif label == b'filenames':
        return filenames
        pass
    else:
        raise NameError


def GetTestDataByLabel(label):
    batch_label = []
    filenames = []

    batch_label.append(unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'batch_label'])
    labels = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'labels']
    data = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'data']
    filenames += unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'filenames']

    label = str.encode(label)
    if label == b'data':
        array = np.ndarray([len(data), 32, 32, 3], dtype=np.int32)
        for i in range(len(data)):
            array[i] = GetPhoto(data[i])
        return array
        pass
    elif label == b'labels':
        return labels
        pass
    elif label == b'batch_label':
        return batch_label
        pass
    elif label == b'filenames':
        return filenames
        pass
    else:
        raise NameError
