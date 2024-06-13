import pickle
import os
import numpy as np
import pandas as pd
import torch
import random
from transformers import BertTokenizer, BertModel


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


def get_data(df, n_distractors, n, tokenizer, model):
    data_word = []
    data_image = []
    data_labels = []

    for _ in range(n):
        samples = df.sample(n=n_distractors+1)

        items_word = samples['label'].to_list()
        with torch.no_grad():
            items_word = get_bert_embedding(items_word, tokenizer, model)
        items_image = samples['image'].to_list()
        labels = random.randrange(n_distractors+1)

        data_word.append(items_word)
        data_image.append(items_image)
        data_labels.append(labels)

    return data_word, data_image, data_labels


def get_npz(df, tokenizer, model, n_distractors=4, n_train=3000, n_test=1000, n_valid=1000):
    print('Making a training set...')
    train_word, train_image, train_labels = get_data(df, n_distractors, n_train, tokenizer, model)
    print('Making a test set...')
    test_word, test_image, test_labels = get_data(df, n_distractors, n_test, tokenizer, model)
    print('Making a validation set...')
    valid_word, valid_image, valid_labels = get_data(df, n_distractors, n_valid, tokenizer, model)

    print('Saving a .npz file for words...')
    file_name = 'data/data_word_' + str(n_distractors) + '_distractors.npz'
    np.savez(file_name,
             train=train_word, train_labels=train_labels,
             valid=valid_word, valid_labels=valid_labels,
             test=test_word, test_labels=test_labels,
             n_distractors=n_distractors)

    print('And one for images...')
    file_name = 'data/data_image_' + str(n_distractors) + '_distractors.npz'
    np.savez(file_name,
             train=train_image, train_labels=train_labels,
             valid=valid_image, valid_labels=valid_labels,
             test=test_image, test_labels=test_labels,
             n_distractors=n_distractors)


if __name__ == "__main__":
    directory = '/Users/dewi-elisa/Documents/Uni/scriptie CLC/Thesis-CLC/Code/data/'

    if os.path.isfile(directory + 'data.pkl'):
        print('Found the file!')
        df = pd.read_pickle(directory + 'data.pkl')
    else:
        print('Preparing dataframe...')
        directory = '/Users/dewi-elisa/Downloads/Imagenet32_train'
        df = get_df(directory)

        directory = '/Users/dewi-elisa/Documents/Uni/scriptie CLC/Thesis-CLC/Code/data/'
        df.to_pickle(directory + 'data.pkl')

    print('Making .npz files...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    get_npz(df, tokenizer, model)
    print('Done!')
