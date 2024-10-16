import pickle
import os
import numpy as np
import pandas as pd
import torch
import random
from transformers import BertTokenizer, BertModel

IMAGE_SIZE = 64

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)

    return dict


def load_data(input_file, image_size):
    d = np.load(input_file)
    x = d['data']
    y = d['labels']

    image_size2 = image_size * image_size

    x = np.dstack((x[:, :image_size2], x[:, image_size2:2*image_size2], x[:, 2*image_size2:]))
    x = x.reshape((x.shape[0], image_size, image_size, 3))

    return x, y


def uint8_to_float32(x):
    return torch.from_numpy(np.array(x)).to(dtype=torch.float) / 255


def get_images(df, n, label):
    if label not in list(df['label']):
        print('not a label!')
        return None

    # get n samples
    label_df = df[df['label'] == label].reset_index()
    images = label_df.sample(n=n, random_state=42)

    # convert from uint8 to float32
    images['image'] = images['image'].apply(uint8_to_float32)

    return images


def get_bert_embedding(sentences, tokenizer, model):
    # Tokenize sentence
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Get last hidden state
    last_hidden_state = outputs.last_hidden_state

    # Get sentence embeddings
    cls_embedding = last_hidden_state[:, 0, :]

    return cls_embedding


def get_df(directory, image_size, tokenizer, model):
    # collect all data
    batch = 'train_data_batch_'
    batches = []

    for i in range(1, 10 + 1):
        batches.append(batch + str(i) + '.npz')

    batch_x = []
    batch_y = []

    for batch in batches:
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
    print('Getting BERT embeddings...')
    with torch.no_grad():
        labels['BERT'] = labels['label'].map(lambda x: get_bert_embedding(x, tokenizer, model)[0])

    # add the labels to the dataframe
    images = images.merge(labels[['id', 'label', 'BERT']], how='left', on='id')

    # replace underscores by spaces
    images['label'] = images['label'].str.lower().replace('_', ' ', regex=True)

    return images


def get_data(df, n_distractors, n):
    data_word = []
    data_image = []
    data_labels = []

    for i in range(n):
        if i % 1000 == 0:
            print(str(i/n * 100) + '%')

        samples = df.sample(n=n_distractors+1)

        # check if all labels are different
        items_word = samples['label'].to_list()
        while len(set(items_word)) != n_distractors+1:
            samples = df.sample(n=n_distractors+1)
            items_word = samples['label'].to_list()

        items_word = samples['BERT'].to_list()
        items_image = samples['image'].to_list()
        labels = random.randrange(n_distractors+1)

        data_word.append(items_word)
        data_image.append(items_image)
        data_labels.append(labels)

    return data_word, data_image, data_labels


def get_npz(df, directory, n_distractors=4, n_train=6000, n_test=2000, n_valid=2000):
    print('Making the data set...')
    words, images, labels = get_data(df, n_distractors, n_train+n_test+n_valid)
    print('Making a training set...')
    train_word, train_image, train_labels = words[:n_train], images[:n_train], labels[:n_train]
    print('Making a test set...')
    test_word, test_image, test_labels = words[n_train:n_train+n_test], images[n_train:n_train+n_test], labels[n_train:n_train+n_test]
    print('Making a validation set...')
    valid_word, valid_image, valid_labels = words[n_train+n_test:], images[n_train+n_test:], labels[n_train+n_test:]

    print('Saving a .npz file for words...')
    file_name = directory + 'data_word_' + str(n_distractors) + '_distractors.npz'
    np.savez(file_name,
             train=train_word, train_labels=train_labels,
             valid=valid_word, valid_labels=valid_labels,
             test=test_word, test_labels=test_labels,
             n_distractors=n_distractors)

    print('And one for images...')
    file_name = directory + 'data_image_' + str(n_distractors) + '_distractors.npz'
    np.savez(file_name,
             train=train_image, train_labels=train_labels,
             valid=valid_image, valid_labels=valid_labels,
             test=test_image, test_labels=test_labels,
             n_distractors=n_distractors)


if __name__ == "__main__":
    directory = 'C:/Users/twank/Documents/Dewi/Thesis-CLC/Code/dataset/'

    if os.path.isfile(directory + 'data.pkl'):
        print('Found the file!')
        df = pd.read_pickle(directory + 'data.pkl')
    else:
        print('Preparing dataframe...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        df = get_df(directory, IMAGE_SIZE, tokenizer, model)

        print('Saving the dataframe...')
        df.to_pickle(directory + 'data.pkl')

    print('Making .npz files...')
    directory = 'C:/Users/twank/Documents/Dewi/Thesis-CLC/Code/data/'
    get_npz(df, directory)
    print('Done!')
