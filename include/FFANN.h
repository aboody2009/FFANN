//Sully Chen 2015

#pragma once
#ifndef FFANN_H
#define FFANN_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "MatrixMath.h"

class FFANN
{
public:
	FFANN(int* dimensions, int num_layers);
	std::vector<Matrix> Weights;
	std::vector<Matrix> Biases;
	int* Dimensions;
	int Num_Layers;
	std::vector<Matrix> FeedForward(Matrix input); //returns the activations of every neuron in each layer
	double TrainWithBackPropagation(Matrix input, Matrix output, double learning_rate);
};

FFANN BreedNetworks(FFANN Parent1, FFANN Parent2, double mutation_probability, double mutation_range);

class RNN
{
public:
    RNN(int input_vector_size, int num_layers);
    std::vector<std::vector<Matrix> > FeedForward(Matrix input, int num_passes); //returns the activations of every neuron in each layer of all passes
    void TrainWithBackPropagation(std::vector<Matrix> sequence, double learning_rate); //trian with backpropagation
    std::vector<Matrix> Weights;
    std::vector<Matrix> Biases;
    std::vector<Matrix> RecurrentWeights;
    int InputVectorSize;
    int Num_Layers;
private:
    std::vector<Matrix> PartialFeedFoward(Matrix input, std::vector<Matrix> recurrences); //used in one time step of the full feedforward pass
    std::vector<Matrix> CalculateInitialDeltas(Matrix output, std::vector<Matrix> outputs); //used in last time step of training
    std::vector<Matrix> CalculateDeltas(Matrix output, std::vector<Matrix> outputs, std::vector<Matrix> nexttimestepdeltas); //used in all time steps except the last time step
};

RNN RNNBreedNetworks(RNN Parent1, RNN Parent2, double mutation_probability, double mutation_range);

#endif
