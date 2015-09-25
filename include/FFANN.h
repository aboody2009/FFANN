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
	float TrainWithBackPropagation(Matrix input, Matrix output, float learning_rate);
};

FFANN BreedNetworks(FFANN Parent1, FFANN Parent2, float mutation_probability, float mutation_range);

class RNN
{
public:
    RNN(int input_vector_size, int num_layers);
    std::vector<std::vector<Matrix> > Feedforward(Matrix input, int num_passes); //returns the activations of every neuron in each layer of all passes
    std::vector<Matrix> Weights;
    std::vector<Matrix> Biases;
    std::vector<Matrix> RecurrentWeights;
    int InputVectorSize;
    int Num_Layers;
};

#endif
