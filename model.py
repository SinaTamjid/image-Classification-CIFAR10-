import tensorflow as tf
from keras.api.layers import Dense,Input,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.api.models import Sequential,Model
from keras.api.datasets import cifar10
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.optimizers import Adam
from keras.api.activations import relu
from keras.api.regularizers import l2

(X_train,y_train),(X_test,y_test)=cifar10.load_data()

X_train=X_train.astype("float32")/255.0
X_test=X_test.astype("float32")/255.0

def my_model():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, 3,padding='same',kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = relu(x)
    x = MaxPooling2D()(x) 
    x = Conv2D(64, 5, padding='same',kernel_regularizer=l2(0.01))(x)  
    x = BatchNormalization()(x)
    x = relu(x)
    x = MaxPooling2D()(x) 
    x = Conv2D(128, 3,padding='same',kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = MaxPooling2D()(x) 
    x = Flatten()(x)
    x = Dense(64, activation='relu',kernel_regularizer=l2(0.01))(x)  
    Dropout(0.5)
    outputs = Dense(10)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model=my_model()


loss=SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss,optimizer=Adam(learning_rate=3e-4),metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=64,epochs=150,verbose=2)
model.evaluate(X_test,y_test,batch_size=64,verbose=2)
print(model.summary())