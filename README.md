# TensorFlow_Fashion_MNIST
Readme for TensorFlow exercises

## Program List with Description
*keras_mnist_fashion.py* - MNIST dataset for simple image classification from TensorFlow tutorial\
                           *Overfitting problem to be solved*\
                           Issue- If you reduce the number of neurons in the dense layer, the training accucracy goes down but the Test accuracy also goes down.\
                                  So it is neccesary to try someother technique.
 \
 \
 *keras_mnist_fashion_overfit_timeDecay.py* - MNIIST Fashion with inverse time deay of learning rate.\
                                              We will see it does not work even for a single hidden layer network.\
                                              Some plotting is also done to familiarize with plotting using the *History* object.
 \
 \
 *keras_mnist_fashion_overfit_solved.py* - L2 Regularization was used to see if the overfitting problem was solvable.\
                                           Though the overfitting to training data redusced, the accuracy with respect to test images continues to remains lower                                                than the accuracy with respect to the training images.\
                                           If we had had more layers of neurons with more neurons L2 regularization might have been enough.\
                                           Nevertheless, we are going to include dropout also.\
                                           So as a next step, L2 + dropout was used.\
                                           The difference in performance between the different implementations can be clearly seen.\
                                           
 *Apart from this another technique one could use more layers of neurons with regularization and check its performance.*
                    
