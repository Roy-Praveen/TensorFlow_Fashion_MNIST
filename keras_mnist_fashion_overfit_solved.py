import tensorflow as tf 
print("tensorFlow version :",tf.__version__)


import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist                                                                   #Invoking module fasion_mnist 

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 
temp = np.array(train_labels)
no_of_datapoints=np.size(temp) 
print(no_of_datapoints)                           

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#matplotlib.pyplot commands to show visualize the image
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

#Dividing by 255 noramilzes the image

train_images = train_images / 255.0

test_images = test_images / 255.0


#View the images
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
''' # No difference observed
'''
plt.figure(figsize=(15,15)) #Size of window
for i in range(25):
    plt.subplot(5,5,i+1)    #Plot location
    plt.xticks([])          #Coordinates of start of plot
    plt.yticks([])          #Coordinates of start of plot
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.hot)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
##############################################################################
'''Building the Model
 The actual layers of neurons are built here
'''
##############################################################################
model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),activity_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(10)
])








#################################################################################################################
'''This is the part where I have introduced decay of the learning rate with time
tf.keras.optimizers.schedule can be used to reduce the learning rate gradually during training'''
#################################################################################################################

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	initial_learning_rate=0.001, decay_steps=no_of_datapoints, decay_rate=0.5,staircase=False)
optimizer = tf.keras.optimizers.Adam(lr_schedule)









###########################################
'''Plotting Runs vs Learning rate Graph'''
###########################################

x_axis = no_of_datapoints*20
step=np.linspace(0.001,x_axis)
lr=lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step,lr)
plt.ylim([0,max(plt.ylim())])
plt.xlim([0,max(plt.xlim())])
plt.xlabel('Runs')
plt.ylabel('Learning Rate')










###########################################################################
'''Specifying the parameters of the model
Parameters such as Loss function, Optimizer and Metrics are specified here
'''
###########################################################################

model1.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,name='Sparse_Categorical_Crossentropy1'),'accuracy'])


model2.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,name='Sparse_Categorical_Crossentropy2'),'accuracy'])




###########################################################################
'''Training the Network
The data to be used to train is fed here'''
###########################################################################


#Model1#
hist1 = model1.fit(train_images, train_labels, epochs=20)
#print(hist1.params)
#print(hist1.history.keys())

Sparse_Categorical_Crossentropy1=hist1.history['Sparse_Categorical_Crossentropy1']




print("The First Model has been fit with no regularization")
test_loss, Sparse_Categorical_Crossentropy, test_acc = model1.evaluate(test_images,  test_labels, verbose=2)
print('\nTest Accuracy (no regularization):', test_acc)
print('\nTest Loss (no regularization):', test_loss)
print('\nSparse_Categorical_Crossentropy (no regularization):', Sparse_Categorical_Crossentropy)



#Model2#
hist2 = model2.fit(train_images, train_labels, epochs=20)
#print(hist2.params)
#print(hist2.history.keys())

Sparse_Categorical_Crossentropy2=hist2.history['Sparse_Categorical_Crossentropy2']

print("The Second Model has been fit with L2 regularization")
test_loss, Sparse_Categorical_Crossentropy, test_acc = model2.evaluate(test_images,  test_labels, verbose=2)
print('\nTest Accuracy (L2 regularization):', test_acc)
print('\nTest Loss (L2 regularization):', test_loss)
print('\nSparse_Categorical_Crossentropy (L2 regularization):', Sparse_Categorical_Crossentropy)





###################################################################################
'''Plotting Cross Entropies vs Runs'''
###################################################################################

step=np.arange(1,21)
fig,ax=plt.subplots()
ax.plot(step,Sparse_Categorical_Crossentropy1,label="Without Regularization")
ax.plot(step,Sparse_Categorical_Crossentropy2,label="With L2 Regularization")
legend=ax.legend(loc='upper right')
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Runs')
plt.ylabel('Sparse_Categorical_Crossentropy')
plt.show()



