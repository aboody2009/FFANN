# FFANN

09/18/2015
This is an easy to use minimal feed forward artificial neural network library. Setup instructions are in the readme!

In your C++ code, include FFANN.h and MatrixMath.h

There is example code in the cpps folder, along with the source code for the libraries in case you want to edit/fix things!

Have fun!

Most Recent Update: 05/19/2016: Hello! It's been a while since I've looked at this code, and I've been doing a lot of projects and research since my last update. Looking at the code with the knowledge I have now, it looks like I severly misunderstood the structures of a lot of the networks I tried to implement almost half a year ago (January). The RNN hidden layers are not connected correctly, resulting in the poor performance and training. The 28x28 convolutional neural network is not connected correctly either, in fact, my code just computes features entirely separate from each other and merges them at the end (I don't know why in the world I expected that to work or make sense).

That being said, I will not be making any updates to this library. It was a good learning experience for me and now that I much greater knowledge of deep learning techniques, structures, and libraries, I don't feel the need to update this code. I am, however, working on another neural network project that I will post soon (hopefully)!

09/19/2015: Support for genetic algorithms has been added!
09/25/2015: Began work on recurrent neural network class, not functional yet
09/26/2015: Recurrent neural network class can do feed forward passes. The back propagation code for the recurrent neural network compiles, but it is messy and does not work. It throws run time errors sometimes, and training typically doesn't work. When a feed forward pass is called on the trained recurrent neural network in Example.cpp, the outputs are strange, and resemble errors I got a while back while working on the FFANN class. These errors were caused by invalid matrices, where there was an incorrect matrix dimension (which should throw a runtime error in the console, weird), or a divide by zero or other mathematical error. I will look into a solution, but this is a tough problem.

11/02/2015: Apologies, it's been a while since I've updated anything. I'm in my junior year of high school so it's getting a little busy. I am working on adding convolutional neural network support, so hopefully something can get working soon. So far, I have not found the problem in the recurrent neural network code.

11/10/2015: I'm having some trouble getting the convolutional neural network code to work, and I'm also having trouble finding ways to make the structure flexible enough. Perhaps I need to do more research into the structure of convolutional neural networks, but either way I hope to find a way to implement the structure soon.

12/05/2015: I added a basic genetic algorithm for recurrent neural networks.

01/18/2016: I added a MNIST dataset test. The MNIST data was preprocessed by someone else and can be found here: http://cis.jhu.edu/~sachin/digit/digit.html

01/22/2016: I added a program that lets the user draw a number using SFML and then attemps to recognize the number using a neural network trained on data from the MNIST dataset

01/24/2016: I think I may have found a fix for the RNN training problem. I'll try to implement it soon, hopefully it works.

01/24/2016: By catching a few errors in the RNN code, as well as changing the training method to update weights and biases at each training step, the RNN training code is now working! Try the example in RNN_Test.cpp! More examples to come soon!

01/27/2016: I added support for a 28x28 convolutional neural network, but it isn't training very well for some reason. I will look into the problem.

05/19/2016: Hello! It's been a while since I've looked at this code, and I've been doing a lot of projects and research since my last update. Looking at the code with the knowledge I have now, it looks like I severly misunderstood the structures of a lot of the networks I tried to implement almost half a year ago (January). The RNN hidden layers are not connected correctly, resulting in the poor performance and training. The 28x28 convolutional neural network is not connected correctly either, in fact, my code just computes features entirely separate from each other and merges them at the end (I don't know why in the world I expected that to work or make sense).

That being said, I will not be making any updates to this library. It was a good learning experience for me and now that I much greater knowledge of deep learning techniques, structures, and libraries, I don't feel the need to update this code. I am, however, working on another neural network project that I will post soon (hopefully)!
