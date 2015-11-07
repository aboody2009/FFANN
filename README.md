# FFANN
09/18/2015
This is an easy to use minimal feed forward artificial neural network library. Setup instructions are in the readme!

In your compiler, use the include folder for the headers, and the lib folder for the libraries
Make sure to link ffann.lib and matrixmath.lib

In your C++ code, include FFANN.h and MatrixMath.h

There is example code in the cpps folder, along with the source code for the libraries in case you want to edit/fix things!

Have fun!

09/19/2015: Support for genetic algorithms has been added! *Note: it has not been tested very much
09/25/2015: Began work on recurrent neural network class, not functional yet
09/26/2015: Recurrent neural network class can do feed forward passes. The back propagation code for the recurrent neural network compiles, but it is messy and does not work. It throws run time errors sometimes, and training typically doesn't work. When a feed forward pass is called on the trained recurrent neural network in Example.cpp, the outputs are strange, and resemble errors I got a while back while working on the FFANN class. These errors were caused by invalid matrices, where there was an incorrect matrix dimension (which should throw a runtime error in the console, weird), or a divide by zero or other mathematical error. I will look into a solution, but this is a tough problem.

11/02/2015: Apologies, it's been a while since I've updated anything. I'm in my junior year of high school so it's getting a little busy. I am working on adding convolutional neural network support, so hopefully something can get working soon. So far, I have not found the problem in the recurrent neural network code.
