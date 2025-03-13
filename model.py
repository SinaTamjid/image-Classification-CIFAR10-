import tensorflow as tf
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical
from keras.api.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.api.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

(X_train,y_train),(X_test,y_test)=cifar10.load_data()

X_train=X_train.astype('float')/255.0
X_test=X_test.astype('float')/255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

model=Sequential()
               #convolutional layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

#connected layer
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100,verbose=2,batch_size=64,validation_data=(X_test,y_test))

#evaluation

loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test accuracy is : {accuracy}")
