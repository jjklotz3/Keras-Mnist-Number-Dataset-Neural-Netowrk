import tensorflow as tf 
import keras 
from keras import layers
import numpy as np 
import matplotlib.pyplot as plt


#Step 1: Load and Process Data

number_dataset = keras.datasets.mnist                               #Load numbers dataset from keras API

(x_train,y_train), (x_test, y_test) = number_dataset.load_data()    #Create out test and train vairbales 
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


class_names = ['0','1','2','3','4','5','6','7','8','9']             #10 availble numbers in the dataset

x_train,x_test = x_train/255,x_test/255                             #Reduce image pixels from RGB to grey scale

#Step 2: Build Model
model = keras.Sequential([                                                                      #Basic CNN for image classification
    keras.layers.Conv2D(32,kernel_size = (3,3),activation = 'relu', input_shape = (28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(rate= 0.25),
    keras.layers.Conv2D(32,kernel_size = (3,3),activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(rate= 0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])   #Setup parameters of the neural network

#Step 3: Train the Model 

history = model.fit(x_train, y_train,                                                               #Train the model. 
                    epochs = 5,
                    batch_size = 32,
                    validation_split = 0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)                                                #Evaluate accuracy and loss function fo the model
print("Test Accuracy: ", test_acc)

#Step 4: Graph and Evaluate Results 
plt.plot(history.history['loss'])             #pulls the metrics we want from model.fit
plt.plot(history.history['val_loss'])         #pulls the metrics we want from model.fit
plt.title('Model Loss')                       #Title 
plt.ylabel('loss')                            #Y axis
plt.xlabel('epoch')                           #X axis
plt.legend(['Train','Validation'],loc='upper left') #Legend 
plt.show()

#Plotting Images and Predictions. Visual Aid 
for i in range(5):                                    #Range is first 5 images in 'test_images'
    plt.grid(False)                                   #We dont want a grid on our graph
    plt.imshow(test_images[i], cmap=plt.cm.binary )   #Show test_images in a form where we can recognize what the image is 
    plt.xlabel("Actual: "+ class_names[test_labels[i]]) #For the x_label, we want to show the Actual Label of the test image and index it with class names so we dont just get a number 
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])]) #For the title, we want to index class names with the number the model gives us and we want to return the nueron with the highest numberic value
    plt.show()

#After evaluation, anything greater than 5 epochs made the model prone to overfitting. Overall Accuracy: 99%