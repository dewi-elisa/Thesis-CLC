import pickle
import numpy as np
import pandas as pd


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_data(input_file, image_size):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    image_size2 = image_size * image_size

    x = np.dstack((x[:, :image_size2], x[:, image_size2:2*image_size2], x[:, 2*image_size2:]))
    x = x.reshape((x.shape[0], image_size, image_size, 3))

    return x, y


def get_images(df, n, label):
    if label not in list(df['label']):
        print('not a label!')
        return None

    label_df = df[df['label'] == label].reset_index()
    print(label_df)
    images = list(label_df['image'].sample(n=n))

    return images


def get_df(directory):
    image_size = 32

    # collect all data
    batch = 'train_data_batch_'
    batches = []

    for i in range(1, 10 + 1):
        batches.append(batch + str(i))

    batch_x = []
    batch_y = []

    for batch in batches:
        print(batch)
        x, y = load_data(directory + '/' + batch, image_size)
        batch_x.append(x)
        batch_y.append(y)

    # flatten data
    flat_x = [x for y in batch_x for x in y]
    flat_y = [x for y in batch_y for x in y]

    # put data in dataframe
    data = {'image': flat_x,
            'id': flat_y}
    images = pd.DataFrame(data)

    # get the labels
    labels = pd.read_csv(directory + '/' + 'map_clsloc.txt', sep=" ", header=None)
    labels.columns = ['wordnet', 'id', 'label']

    # add the labels to the dataframe
    images = images.merge(labels[['id', 'label']], how='outer', on='id')

    # replace underscores by spaces
    images['label'] = images['label'].str.lower().replace('_', ' ', regex=True)

    return images


if __name__ == "__main__":
    directory = '/Users/dewi-elisa/Downloads/Imagenet32_train'

    images = get_df(directory)
    goose_images = get_images(images, 10, 'goose')

    print(images['label'].value_counts())
