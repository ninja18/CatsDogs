from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
folders = ['train','test']
# enumerate files in the directory
for folder in folders:
    photos, labels = list(), list()
    for file in listdir(folder):
        # determine class
        output = 0.0
        if file.startswith('cat'):
            output = 1.0
        # load image
        photo = load_img(folder + '/' + file, target_size=(200, 200))
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)
    # convert to a numpy arrays
    photos = asarray(photos)
    labels = asarray(labels)
    print(photos.shape, labels.shape)
    # save the reshaped photos
    save(folder+'-dogs_vs_cats_photos.npy', photos)
    save(folder+'-dogs_vs_cats_labels.npy', labels)
