# cnn

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
# --------------- Preprocessing Database -----------------------------
def extract_label():
    complete_text = open("C:\\Users\\Prabha\\Desktop\\My codes\\cnn\\database\\information\\Info.txt")
    lines = [line.rstrip('\n') for line in complete_text]

    details = {}
    for line in lines:
        value = line.split(' ')
        #print(value)
        if len(value)> 3:
            if(value[3]=='B'):
                details[value[0]] = 0
            if(value[3]=='M'):
                details[value[0]] = 1
    return details

def extract_image():
    details = {}

    for i in range(322):
        path = "C:\\Users\\Prabha\\Desktop\\My codes\\cnn\\database\\images"
        if i<9:
            path = path + '\\mdb00'+ str(i+1)+".pgm"
            filelabel = 'mdb00'+ str(i+1)
        elif i<99:
            path = path + '\\mdb0'+ str(i+1)+".pgm"
            filelabel = 'mdb0'+ str(i+1)
        else :
            path =path + '\\mdb' + str(i+1)+".pgm"
            filelabel = 'mdb' + str(i+1)

        #i = i+1
        image = cv.imread(path, -1)
        #print(path)
        image = cv.resize(image, (64,64))
        details[filelabel] = image

    return details


# --------------- Creating the test, train split ----------------------

def spl():
    labels = extract_label()
    images = extract_image()
    imageids = labels.keys()

    X = []
    Y = []

    for id in imageids:
        X.append(images[id])
        Y.append(labels[id])

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
    a,b,c=X_train.shape  # (60000, 28, 28)
    X_train = np.reshape(X_train, (a, b, c, 1))  #1 for gray scale
    a, b, c=X_test.shape
    X_test = np.reshape(X_test, (a, b, c, 1))   #1 for gray scale
    return(X_train, Y_train, X_test, Y_test)

# ----------------- CNN Model -----------------------------------------
def cnn(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3), activation= 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=15, batch_size=128)
    loss, accuracy = model.evaluate(x_train, y_train)

    probabilities = model.predict(x_test)
    predictions = [float(np.round(x)) for x in probabilities]
    accuracy = np.mean(predictions == y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy*100))
    model.save('result.h5')


# ------------------- Main function ------------------------------

X_train, Y_train, X_test, Y_test = spl()
cnn(X_train, Y_train, X_test, Y_test)
