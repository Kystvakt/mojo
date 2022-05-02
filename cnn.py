from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import NASNetMobile, ResNet50V2, VGG19, EfficientNetB4
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
# import tensorflow_addons as tfa
# import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
import os
from datetime import datetime, timezone, timedelta


### Arguments ###

parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
parser.add_argument("model", choices=["NASNet", "ResNet", "VGGNet", "EfficientNet"], help="model")
parser.add_argument("--size", "-s", type=int, default=224, help="image size")
parser.add_argument(
    "--optimizer", "-o",
    choices=["adam", "nadam", "adadelta", "RMSprop"],
    default="adam",
    help="optimizer",
)
parser.add_argument(
    "--learning_rate", "-l", type=float, default=5e-6, help="learning rate"
)
parser.add_argument(
    "--activation", "-a",
    choices=["relu", "elu", "softplus"],
    default="relu",
    help="activation function"
)
parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size")
parser.add_argument("--epochs", "-e", type=int, default=100, help="epochs")
parser.add_argument("--visualize", "-v", action="store_true", default=False, help="visualize")
parser.add_argument("--fileout", "-f", action="store_true", default=False, help="save as csv file")
parser.add_argument("--model_save", "-m", action="store_true", default=False, help="save model as .h5 file")
args = parser.parse_args()


### Datasets ###

train_dir = "Korea_food/train"
valid_dir = "Korea_food/test"
test_dir = "Korea_food/test"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

target_size = (args.size, args.size)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=args.batch_size, class_mode="categorical"
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=target_size, batch_size=args.batch_size, class_mode="categorical"
)


### Model DEfine ###

input_shape = (args.size, args.size, 3)
if args.model == 'NASNet':
    conv_base = NASNetMobile(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
elif args.model == 'ResNet':
    conv_base = ResNet50V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
elif args.model == 'VGGNet':
    conv_base = VGG19(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
elif args.model == 'EfficientNet':
    conv_base = EfficientNetB4(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dense(256, activation=args.activation))
model.add(layers.Dense(64, activation=args.activation))
model.add(layers.Dense(30, activation="softmax"))
model.summary()

if args.optimizer == 'adam':
    optim = optimizers.Adam(learning_rate=args.learning_rate)
elif args.optimizer == 'nadam':
    optim = optimizers.Nadam(learning_rate=args.learning_rate)
elif args.optimizer == 'adadelta':
    optim = optimizers.Adadelta(learning_rate=args.learning_rate)
elif args.optimizer == 'RMSprop':
    optim = optimizers.RMSprop(learning_rate=args.learning_rate)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optim,
    metrics=["accuracy"],
)


### Model Train ###

# Callbacks
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

time_callback = TimeHistory()
# tqdm_callback = tfa.callbacks.TQDMProgressBar()
es_callback = EarlyStopping(monitor='loss', patience=5)


# Train
model_hist = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=args.epochs,
    validation_data=valid_generator,
    validation_steps=50,
    verbose=1,
    callbacks=[
        time_callback,
        # tqdm_callback,
        # es_callback,
        # cp_callback,
    ]
)
endtime = datetime.now().strftime("%Y%m%d%H%M%S")

# Save as CSV file
if args.fileout:
    hist_df = pd.DataFrame(model_hist.history)
    hist_df["time"] = pd.Series(time_callback.times)
    elements = "_".join([
        args.model,
        str(args.size),
        args.optimizer,
        str(args.learning_rate),
        args.activation,
        str(args.batch_size),
        endtime,
    ])
    filename = "result/" + elements + ".csv"
    with open(filename, 'w') as f:
        hist_df.to_csv(f)


### Visualize ###

if args.visualize:
    epoch_length = len(time_callback.times)
    runtime = [sum(time_callback.times[:x]) for x in range(epoch_length)]
    labels = ["Training", "Validation"]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].plot(runtime, model_hist.history["loss"])
    ax[0].plot(runtime, model_hist.history["val_loss"])
    ax[0].set_title("Loss")
    ax[0].set_xlabel(f"Runtime (s) for {epoch_length} epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(labels)
    ax[0].grid(alpha=0.2, zorder=3)

    ax[1].plot(runtime, model_hist.history["accuracy"])
    ax[1].plot(runtime, model_hist.history["val_accuracy"])
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel(f"Runtime (s) for {epoch_length} epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(labels)
    ax[1].grid(alpha=0.2, zorder=3)

    plt.tight_layout()
    plt.show()


### Evaluate ###

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=args.batch_size,
    class_mode="categorical"
)
loss, accuracy = model.evaluate(test_generator, steps=50)
print(f"Loss      {loss:>8.5f}")
print(f"Accuracy  {accuracy:>8.5f}")


### Model Save ###

if args.model_save:
    elements = "_".join([
        args.model,
        str(args.size),
        args.optimizer,
        str(args.learning_rate),
        args.activation,
        str(args.batch_size),
        endtime,
    ])
    model_file_name = elements + '.h5'
    save_path = 'model/'
    model.save(os.path.join(save_path, model_file_name))
