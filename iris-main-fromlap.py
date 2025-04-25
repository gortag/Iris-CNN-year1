import os
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
import PIL
import random
import math
import numpy as np
import cv2
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.optimizers import Adam
from keras.api.layers import BatchNormalization, Input, Conv2D, GaussianNoise,MaxPooling2D, Flatten, Dense, Dropout

def load_dataset(path):

    labels = []
    images = []

    for folder in os.listdir(path):
        for lr in os.listdir(path + '/' + folder):
            for image in os.listdir(path + '/' + folder + '/' + lr):
                if image.endswith('b') is False:
                    images.append(path + '/' + folder + '/' + lr + '/' + image)
                    labels.append(folder + '-' + lr)

    df = pd.DataFrame(list(zip(labels, images)), columns=['Label', 'ImagePath'])
    return df, labels, images

df, labels, images = load_dataset('C:/casia/CASIA-Iris-Thousand/CASIA-Iris-Thousand')

def explore_data(df):

    head = pd.DataFrame(df.head())
    tail = pd.DataFrame(df.tail())
    nunique = pd.DataFrame(df.nunique(), columns=["#_of_Unique"])
    describe = pd.DataFrame(df.describe())
    dtypes =  pd.DataFrame(df.dtypes, columns=["Datatype"])
    labels_distribution = pd.DataFrame(df['Label'].value_counts())
    results = {
        'Table 2: Dataset head:':head,
        'Table 3: Dataset tail:':tail,
        'Table 4: Dataset numerical description: ':describe,
        'Table 6: Dataset columns data types: ':dtypes,
        'Table 7: Number of unique images in the datasets:':nunique,
        'Table 8: Labels distribution:':labels_distribution}
    return results

def print_dataset_exploration(results):

    for operation, dataframe in results.items():
        print(f"{operation}")
        if operation == 'Table 6: Missing Values By Percentage':
            print("Total Sum of Missing Percentage: ", dataframe['Percentage'].sum())
        display(dataframe)

print_dataset_exploration(explore_data(df))


def show_random_samples(df, num):

    random.seed(1190652)
    random_indices = random.sample(range(df.shape[0]), num)
    num_rows = math.ceil(num / 4)

    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
    for i, idx in enumerate(random_indices):
        row = i // 4
        col = i % 4

        if idx < df.shape[0]:
            image_path = df.loc[idx, "ImagePath"]
            image = PIL.Image.open(image_path)
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Image {idx} Person Label: {df.loc[idx, 'Label']}")
            ax.axis("off")

    plt.suptitle("Figure 3: Random Small Sample of the Dataset")
    plt.tight_layout()
    plt.show()

SIZE = 20000
NUM_CLASSES = 2000
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CHANNELS = 1
input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

def resize(img, target_height=IMG_HEIGHT, target_width=IMG_WIDTH, pad_value=255):

    aspect_ratio = img.shape[1] / img.shape[0]

    if aspect_ratio > target_width / target_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))

    preprocessed_img = np.full((target_height, target_width), pad_value, dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    preprocessed_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return preprocessed_img

def preprocess_image(img_dir):

    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    img = resize(img)
    img = img / 255.

    return img

plt.imshow(preprocess_image(df.iloc[987]['ImagePath']), cmap="gray")
plt.title("Figure 8: Preprocessed image sample")
plt.show()

def preprocess_labels(df):

    labels = df['Label'].astype(str)
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    return labels

print("Label after encoding: ", preprocess_labels(df)[11])
print("Label before encoding: ",df.iloc[11]['Label'])


def split_dataset(preprocessed_images, preprocessed_labels, train_size=0.8, validation_size=0.1, shuffle=True):

    np.random.seed(1190652)
    indices = np.arange(SIZE)
    if shuffle:
        np.random.shuffle(indices)

    train_samples = int(SIZE * train_size)
    validation_samples = int(SIZE * validation_size)

    train_indices = indices[:train_samples]
    validation_indices = indices[train_samples:train_samples + validation_samples]
    test_indices = indices[train_samples + validation_samples:]

    x_train = preprocessed_images[train_indices]
    y_train = preprocessed_labels[train_indices]
    x_valid = preprocessed_images[validation_indices]
    y_valid = preprocessed_labels[validation_indices]
    x_test = preprocessed_images[test_indices]
    y_test = preprocessed_labels[test_indices]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def prepare_dataset(df):

    preprocessed_images = []
    for i in range(SIZE):
        image = preprocess_image(images[i])
        preprocessed_images.append(image)

    preprocessed_images = np.array(preprocessed_images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
    preprocessed_labels = preprocess_labels(df)
    return split_dataset(preprocessed_images, preprocessed_labels)


x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_dataset(df)
print("Training set size: ", x_train.shape)
print("Validation set size: ", x_valid.shape)
print("Testing set size: ", x_test.shape)

EPOCHS = 100
BATCH_SIZE = 32
loss = 'sparse_categorical_crossentropy'
activation = "leaky_relu"
initial_learning_rate = 0.001
optimizer = Adam(learning_rate=initial_learning_rate)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


def create_model():

    padding = 'same'
    poolpadding = 'valid'

    model = Sequential([
        Input(shape = input_shape),

        Conv2D(32, (5, 5), padding=padding, activation=activation, name="Conv1"),
        BatchNormalization(axis=-1, name="BN1"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool1"),
        GaussianNoise(0.1, name="GaussianNoise"),
        Dropout(0.1, name="Dropout1"),

        Conv2D(64, (5, 5), padding=padding, activation=activation, name="Conv2"),
        BatchNormalization(axis=-1, name="BN2"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool2"),
        Dropout(0.1, name="Dropout2"),

        Conv2D(128, (5, 5), padding=padding, activation=activation, name="Conv3"),
        BatchNormalization(axis=-1, name="BN3"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool3"),
        Dropout(0.25, name="Dropout3"),

        Conv2D(256, (3, 3), padding=padding, activation=activation, name="Conv4"),
        BatchNormalization(axis=-1, name="BN4"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool4"),
        Dropout(0.25, name="Dropout4"),

        Conv2D(256, (3, 3), padding=padding, activation=activation, name="Conv5"),
        BatchNormalization(axis=-1, name="BN5"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool5"),
        Dropout(0.25, name="Dropout5"),

        Conv2D(512, (3, 3), padding=padding, activation=activation, name="Conv6"),
        BatchNormalization(axis=-1, name="BN6"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool6"),
        Dropout(0.45, name="Dropout6"),

        Conv2D(512, (2, 2), padding=padding, activation=activation, name="Conv7"),
        BatchNormalization(axis=-1, name="BN7"),
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool7"),
        Dropout(0.5, name="Dropout7"),

        Flatten(),
        Dense(128, activation=activation, name="Dense1"),
        Dense(2000, activation='softmax', name="SoftmaxClasses"),
    ],
        name="iris-bio-model")
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

model = create_model()
model.summary()

history = model.fit(np.array(x_train), y_train, validation_data=(np.array(x_valid), y_valid),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[earlyStopping, reduce_lr_loss])

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Figure 10: Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1, len(training_loss) + 1)

plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title('Figure 11: Loss Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("iris-bio-model.h5")