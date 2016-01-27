//Sully Chen 2015

#pragma once
#ifndef FFANN_H
#define FFANN_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include "MatrixMath.h"

struct BackpropagationData
{
    double cost;
    std::vector<Matrix> activations;
    std::vector<Matrix> deltas;
};

class FFANN
{
public:
    FFANN();
    FFANN(int* dimensions, int num_layers);
    std::vector<Matrix> Weights;
    std::vector<Matrix> Biases;
    int* Dimensions;
    int Num_Layers;
    std::vector<Matrix> FeedForward(Matrix input); //returns the activations of every neuron in each layer
    double TrainWithBackPropagation(Matrix input, Matrix output, double learning_rate);
    double TrainWithBackPropagation(Matrix input, Matrix output, std::vector<Matrix> outputs, double learning_rate, Matrix* FirstLayerDeltas); //used for CovNet, returns the delta values of the first layer;
};

FFANN BreedNetworks(FFANN Parent1, FFANN Parent2, double mutation_probability, double mutation_range);

class RNN
{
public:
    RNN(int input_vector_size, int num_layers);
    std::vector<std::vector<Matrix> > FeedForward(Matrix input, int num_passes); //returns the activations of every neuron in each layer of all passes
    double TrainWithBackPropagation(std::vector<Matrix> sequence, double learning_rate); //trian with backpropagation
    std::vector<Matrix> Weights;
    std::vector<Matrix> Biases;
    std::vector<Matrix> RecurrentWeights;
    int InputVectorSize;
    int Num_Layers;
private:
    std::vector<Matrix> PartialFeedFoward(Matrix input, std::vector<Matrix> recurrences); //used in one time step of the full feedforward pass
    BackpropagationData TrainStep(Matrix input, Matrix output, std::vector<Matrix> recurrences);
};

RNN RNNBreedNetworks(RNN Parent1, RNN Parent2, double mutation_probability, double mutation_range);

//COVNET NOT YET WORKING PROPERLY
class CovNet28x28 //28x28 Layer1 ----5x5 Kernel---> 24x24 Layer2 ----Max Pooling---> 12x12 Layer3 ----3x3 Kernel---> 10x10 Layer 4 ---fully connected---> 10 output layer
{
public:
    CovNet28x28();
    Matrix Layer1_Bias; //28x28
    std::vector<Matrix> Kernels1; //5x5, 9 kernels
    std::vector<Matrix> Layer3_Biases; //12x12
    std::vector<Matrix> Kernels2; //3x3, 9 kernels
    FFANN FullyConnectedLayer; //900 - 10
    std::vector<std::vector<Matrix> > FeedForward(Matrix input);
    double TrainWithBackPropagation(Matrix input, Matrix output, double learning_rate);
};

int MaxElement(Matrix m);

#endif
