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
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
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

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,name='Sparse_Categorical_Crossentropy'),'accuracy'])







###########################################################################
'''Training the Network
The data to be used to train is fed here'''
###########################################################################

hist = model.fit(train_images, train_labels, epochs=20)

print(hist.params)

print(hist.history.keys())

Sparse_Categorical_Crossentropy=hist.history['Sparse_Categorical_Crossentropy']


step=np.arange(1,21)
print (step)

plt.figure(figsize = (8,6))
plt.plot(step,Sparse_Categorical_Crossentropy)
a = plt.xscale('log') #Log scale. Change it if required
plt.xlim([0,max(plt.xlim())])
plt.xlabel('Runs')
plt.ylabel('Sparse_Categorical_Crossentropy')




###########################################################################
'''Accuracy Evaluation
 The loss is reduced and the accuracy is calculated (I would rather call it "fit" than accuracy)
	But it has to be tested with new data
	Remember it is this analysis which makes the model valid
'''
###########################################################################

test_loss, Sparse_Categorical_Crossentropy ,test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest Accuracy:', test_acc)
print('\nTest Loss:', test_loss)
print('\nSparse_Categorical_Crossentropy:', Sparse_Categorical_Crossentropy)


















#Predicting using the model
'''The final output comes as a vactor for every training data point
	we calculate the prediction precision using a probailities
'''
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])  #Adding another softmax layer to the outputs to convert them to probailities

predictions = probability_model(test_images) #Making this model make predictions

predictions[0]
predictions[1]
predictions[2]
print(".....")


np.argmax(predictions[0])
test_labels[0] #The prediction and the actual label should match


'''
	Graphs to analyze prediction on the test images
'''

def plot_image(i, predictions_array, true_label, img):
	true_label, img = true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color= 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)



def plot_value_array(i, predictions_array, true_label):
	true_label = true_label[i]
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')


'''
	Using these defined plotting functions to see how are results look
'''

#First Test Image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#Twelveth Test Image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#Hundredth Test Image
i = 100
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#5000th Test image
i = 5000
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


#Finally we are going to use the model to classify a single image
#This is also from the test set
#But it is treated as a real world example

img = test_images[1]
print(img.shape)


'''
	tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.
	Accordingly, even though you're using a single image, you need to add it to a list:
'''

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45) #"_" means a variable we dont care about
plt.show()

#tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data.
#Grab the predictions for our (only) image in the batch:

print(np.argmax(predictions_single[0]))
