//Sully Chen 2015

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include "MatrixMath.h"
#include "FFANN.h"

FFANN::FFANN()
{
}

FFANN::FFANN(int* dimensions, int num_layers)
{
    Dimensions = dimensions;
    Num_Layers = num_layers;
    
    //the first elements of the weights matrix vector object is just a filler to make the math look cleaner, it's not actually used
    Matrix temp;
    Weights.push_back(temp);
    
    //create randomized weight matrices
    for (int i = 0; i < num_layers - 1; i++)
    {
        Matrix m(dimensions[i], dimensions[i + 1]);
        for (int j = 0; j < dimensions[i]; j++)
        {
            for (int k = 0; k < dimensions[i + 1]; k++)
            {
                m.Elements[j * m.Dimensions[1] + k] = (rand() % 200 - 100) / 1000.0f;
            }
        }
        Weights.push_back(m);
    }
    
    //create biases
    for (int i = 0; i < num_layers; i++)
    {
        Matrix m(dimensions[i], 1);
        for (int j = 0; j < dimensions[i]; j++)
        {
            m.Elements[j] = (rand() % 200 - 100) / 1000.0f;
        }
        Biases.push_back(m);
    }
}

std::vector<Matrix> FFANN::FeedForward(Matrix input)
{
    std::vector<Matrix> outputs;
    //Add biases and apply activation function to each input element
    for (int i = 0; i < input.Dimensions[0]; i++)
    {
        input.Elements[i] += Biases[0].Elements[i];
        input.Elements[i] = 1 / (1 + pow(2.718281828459f, -input.Elements[i]));
    }
    outputs.push_back(input);
    
    //feed forward calculation
    for (int i = 1; i < Num_Layers; i++)
    {
        //feed forward
        Matrix z;
        z = Weights[i].Transpose() * outputs[i - 1] + Biases[i];
        
        outputs.push_back(z);
        
        //Apply activation function
        for (int j = 0; j < outputs[i].Dimensions[0]; j++)
        {
            outputs[i].Elements[j] = 1 / (1 + pow(2.718281828459f, -outputs[i].Elements[j]));
        }
    }
    
    return outputs;
}

double FFANN::TrainWithBackPropagation(Matrix input, Matrix output, double learning_rate)
{
    std::vector<Matrix> outputs = FeedForward(input);
    
    std::vector<Matrix> temp_deltas; //layer deltas stored backwards in order
    
    //calculate cost function
    double cost = 0.0f;
    Matrix partial_cost_matrix(Dimensions[Num_Layers - 1], 1);
    partial_cost_matrix = output + (outputs[outputs.size() - 1] * -1);
    for (int i = 0; i < partial_cost_matrix.Elements.size(); i++)
    {
        cost += 0.5f * partial_cost_matrix.Elements[i] * partial_cost_matrix.Elements[i];
    }
    //calculate last layer deltas
    Matrix lld(Dimensions[Num_Layers - 1], 1);
    lld = outputs[outputs.size() - 1] + (output * -1);
    for (int i = 0; i < lld.Dimensions[0]; i++)
    {
        double a = outputs[outputs.size() - 1].Elements[i];
        lld.Elements[i] *= a * (1 - a); //derivative of activation function
    }
    temp_deltas.push_back(lld);
    
    //calculate the rest of the deltas through back propagation
    int j = 0; //this keeps track of the index for the next layer's delta
    for (int i = Num_Layers - 2; i >= 0; i--) //start at the second to last layer
    {
        Matrix delta(Dimensions[i], 1);
        delta = Weights[i + 1] * temp_deltas[j];
        j++;
        for (int k = 0; k < delta.Dimensions[0]; k++)
        {
            double a = outputs[i].Elements[k];
            delta.Elements[k] *= a * (1 - a); //derivative of activation function
        }
        temp_deltas.push_back(delta);
    }
    
    //put the deltas into a new vector object in the correct order
    std::vector<Matrix> deltas;
    for (int i = (int)temp_deltas.size() - 1; i >= 0; i--)
    {
        deltas.push_back(temp_deltas[i]);
    }
    
    //update biases
    for (int i = 0; i < Biases.size(); i++)
    {
        Biases[i] = Biases[i] + deltas[i] * (-1.0f * learning_rate);
    }
    
    //update weights
    for (int i = 1; i < Weights.size(); i++)
    {
        Weights[i] = Weights[i] + ((outputs[i - 1] * deltas[i].Transpose()) * (-1.0f * learning_rate));
    }
    
    return cost;
}

double FFANN::TrainWithBackPropagation(Matrix input, Matrix output, std::vector<Matrix> outputs, double learning_rate, Matrix* FirstLayerDeltas) //used for CovNet, returns the delta values of the first layer;
{
    std::vector<Matrix> temp_deltas; //layer deltas stored backwards in order
    
    //calculate cost function
    double cost = 0.0f;
    Matrix partial_cost_matrix(Dimensions[Num_Layers - 1], 1);
    partial_cost_matrix = output + (outputs[outputs.size() - 1] * -1);
    for (int i = 0; i < partial_cost_matrix.Elements.size(); i++)
    {
        cost += 0.5f * partial_cost_matrix.Elements[i] * partial_cost_matrix.Elements[i];
    }
    //calculate last layer deltas
    Matrix lld(Dimensions[Num_Layers - 1], 1);
    lld = outputs[outputs.size() - 1] + (output * -1);
    for (int i = 0; i < lld.Dimensions[0]; i++)
    {
        double a = outputs[outputs.size() - 1].Elements[i];
        lld.Elements[i] *= a * (1 - a); //derivative of activation function
    }
    temp_deltas.push_back(lld);
    
    //calculate the rest of the deltas through back propagation
    int j = 0; //this keeps track of the index for the next layer's delta
    for (int i = Num_Layers - 2; i >= 0; i--) //start at the second to last layer
    {
        Matrix delta(Dimensions[i], 1);
        delta = Weights[i + 1] * temp_deltas[j];
        j++;
        for (int k = 0; k < delta.Dimensions[0]; k++)
        {
            double a = outputs[i].Elements[k];
            delta.Elements[k] *= a * (1 - a); //derivative of activation function
        }
        temp_deltas.push_back(delta);
    }
    
    //put the deltas into a new vector object in the correct order
    std::vector<Matrix> deltas;
    for (int i = (int)temp_deltas.size() - 1; i >= 0; i--)
    {
        deltas.push_back(temp_deltas[i]);
    }
    
    //update biases
    for (int i = 0; i < Biases.size(); i++)
    {
        Biases[i] = Biases[i] + deltas[i] * (-1.0f * learning_rate);
    }
    
    //update weights
    for (int i = 1; i < Weights.size(); i++)
    {
        Weights[i] = Weights[i] + ((outputs[i - 1] * deltas[i].Transpose()) * (-1.0f * learning_rate));
    }
    
    *FirstLayerDeltas = deltas[0]; //save first layer deltas
    
    return cost;
}

FFANN BreedNetworks(FFANN Parent1, FFANN Parent2, double mutation_probability, double mutation_range)
{
    if (mutation_probability > 1.0f)
    {
        mutation_probability = 1.0f;
        std::cout << "Warning: keep mutation probability between 0.0 and 1.0. Capping to 1.0" << std::endl;
    }
    else if (mutation_probability < 0.0f)
    {
        mutation_probability = 0.0f;
        std::cout << "Warning: keep mutation probability between 0.0 and 1.0. Flooring to 0.0" << std::endl;
    }
    //Make sure the networks are the same size
    if (Parent1.Num_Layers != Parent2.Num_Layers)
    {
        std::cout << "Error! Cannot breed due to network formating!" << std::endl;
        FFANN ffann(0, 0);
        return ffann;
    }
    for (int i = 0; i < Parent1.Num_Layers; i++)
    {
        if (Parent1.Dimensions[i] != Parent2.Dimensions[i])
        {
            std::cout << "Error! Cannot breed due to network formating!" << std::endl;
            FFANN ffann(0, 0);
            return ffann;
        }
    }
    
    //Genetic algorithm
    FFANN offspringnetwork(Parent1.Dimensions, Parent1.Num_Layers); //initialize offspring network
    
    //crossover the genes of the weights
    for (int i = 1; i < Parent1.Num_Layers; i++) //we start at 1 because weights[0] is a filler matrix and does not contain any elements
    {
        int crossoverpoint = rand() % Parent1.Weights[i].Elements.size();
        for (int j = 0; j < crossoverpoint; j++)
        {
            offspringnetwork.Weights[i].Elements[j] = Parent1.Weights[i].Elements[j]; //one part of the gene is from parent 1
        }
        for (int j = crossoverpoint; j < offspringnetwork.Weights[i].Elements.size(); j++)
        {
            offspringnetwork.Weights[i].Elements[j] = Parent2.Weights[i].Elements[j]; //the other part of the gene is from parent 2
        }
        
        //randomly mutate genes
        for (int k = 0; k < offspringnetwork.Weights[i].Elements.size(); k++)
        {
            int random_int = rand() % (int)((1.01f - mutation_probability) * 1000); //we round mutation_probability * 1000 to an integer to ensure it is not a double
            for (int j = 0; j < 10; j++) //we're choosing out of 1000 to get precision up to the hundredth place, so we must take 10 samples to get a probability out of 100
            {
                //random selection of gene
                if (random_int == rand() % (int)(mutation_probability * 1000))
                {
                    offspringnetwork.Weights[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
                }
            }
        }
    }
    
    //crossover the genes of the biases
    for (int i = 0; i < Parent1.Num_Layers; i++)
    {
        int crossoverpoint = rand() % Parent1.Biases[i].Elements.size();
        for (int j = 0; j < crossoverpoint; j++)
        {
            offspringnetwork.Biases[i].Elements[j] = Parent1.Biases[i].Elements[j]; //one part of the gene is from parent 1
        }
        for (int j = crossoverpoint; j < offspringnetwork.Biases[i].Elements.size(); j++)
        {
            offspringnetwork.Biases[i].Elements[j] = Parent2.Biases[i].Elements[j]; //the other part of the gene is from parent 2
        }
        
        //randomly mutate genes
        for (int k = 0; k < offspringnetwork.Biases[i].Elements.size(); k++)
        {
            int random_int = rand() % (int)((1.01f - mutation_probability) * 1000); //we round mutation_probability * 1000 to an integer to ensure it is not a double
            for (int j = 0; j < 10; j++) //we're choosing out of 1000 to get precision up to the hundredth place, so we must take 10 samples to get a probability out of 100
            {
                //random selection of gene
                if (random_int == rand() % (int)(mutation_probability * 1000))
                {
                    offspringnetwork.Biases[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
                }
            }
        }
    }
    
    return offspringnetwork;
}

//This RNN code is not implemented correctly and does not function correctly
/*
RNN::RNN(int input_vector_size, int num_layers) : InputVectorSize(input_vector_size), Num_Layers(num_layers)
{
    //the first elements of the weights matrix vector object is just a filler to make the math look cleaner, it's not actually used
    Matrix temp;
    Weights.push_back(temp);
    
    //create randomized weight matrices
    for (int i = 0; i < num_layers - 1; i++)
    {
        Matrix m(input_vector_size, input_vector_size);
        for (int j = 0; j < input_vector_size; j++)
        {
            for (int k = 0; k < input_vector_size; k++)
            {
                m.Elements[j * m.Dimensions[1] + k] = (rand() % 200 - 100) / 1000.0f;
            }
        }
        Weights.push_back(m);
    }
    
    //create recurrent weights
    //the first and last recurrent weight matrices are placeholders, they are not used
    for (int i = 0; i < num_layers; i++)
    {
        Matrix m(input_vector_size, 1);
        for (int j = 0; j < input_vector_size; j++)
        {
            m.Elements[j] = (rand() % 200 - 100) / 1000.0f;
        }
        RecurrentWeights.push_back(m);
    }
    
    //create biases
    for (int i = 0; i < num_layers; i++)
    {
        Matrix m(input_vector_size, 1);
        for (int j = 0; j < input_vector_size; j++)
        {
            m.Elements[j] = (rand() % 200 - 100) / 1000.0f;
        }
        Biases.push_back(m);
    }
}

RNN RNNBreedNetworks(RNN Parent1, RNN Parent2, double mutation_probability, double mutation_range)
{
    if (mutation_probability > 1.0f)
    {
        mutation_probability = 1.0f;
        std::cout << "Warning: keep mutation probability between 0.0 and 1.0. Capping to 1.0" << std::endl;
    }
    else if (mutation_probability < 0.0f)
    {
        mutation_probability = 0.0f;
        std::cout << "Warning: keep mutation probability between 0.0 and 1.0. Flooring to 0.0" << std::endl;
    }
    //Make sure the networks are the same size
    if (Parent1.Num_Layers != Parent2.Num_Layers)
    {
        std::cout << "Error! Cannot breed due to network formating!" << std::endl;
        RNN rnn(0, 0);
        return rnn;
    }
    if (Parent1.Num_Layers != Parent2.Num_Layers || Parent1.InputVectorSize != Parent2.InputVectorSize)
    {
        std::cout << "Error! Cannot breed due to network formating!" << std::endl;
        RNN rnn(0, 0);
        return rnn;
    }
    
    //Genetic algorithm
    RNN offspringnetwork(Parent1.InputVectorSize, Parent1.Num_Layers); //initialize offspring network
    
    //crossover the genes of the weights
    for (int i = 1; i < Parent1.Num_Layers; i++) //we start at 1 because weights[0] is a filler matrix and does not contain any elements
    {
        int crossoverpoint = rand() % Parent1.Weights[i].Elements.size();
        for (int j = 0; j < crossoverpoint; j++)
        {
            offspringnetwork.Weights[i].Elements[j] = Parent1.Weights[i].Elements[j]; //one part of the gene is from parent 1
        }
        for (int j = crossoverpoint; j < offspringnetwork.Weights[i].Elements.size(); j++)
        {
            offspringnetwork.Weights[i].Elements[j] = Parent2.Weights[i].Elements[j]; //the other part of the gene is from parent 2
        }
        
        //randomly mutate genes
        for (int k = 0; k < offspringnetwork.Weights[i].Elements.size(); k++)
        {
            int random_int = rand() % (int)((1.01f - mutation_probability) * 1000); //we round mutation_probability * 1000 to an integer to ensure it is not a double
            for (int j = 0; j < 10; j++) //we're choosing out of 1000 to get precision up to the hundredth place, so we must take 10 samples to get a probability out of 100
            {
                //random selection of gene
                if (random_int == rand() % (int)(mutation_probability * 1000))
                {
                    offspringnetwork.Weights[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
                }
            }
        }
    }
    
    //crossover the genes of the biases and recurrent weights
    for (int i = 0; i < Parent1.Num_Layers; i++)
    {
        int crossoverpoint = rand() % Parent1.Biases[i].Elements.size();
        for (int j = 0; j < crossoverpoint; j++)
        {
            offspringnetwork.Biases[i].Elements[j] = Parent1.Biases[i].Elements[j]; //one part of the gene is from parent 1
            offspringnetwork.RecurrentWeights[i].Elements[j] = Parent1.RecurrentWeights[i].Elements[j];
        }
        for (int j = crossoverpoint; j < offspringnetwork.Biases[i].Elements.size(); j++)
        {
            offspringnetwork.Biases[i].Elements[j] = Parent2.Biases[i].Elements[j]; //the other part of the gene is from parent 2
            offspringnetwork.RecurrentWeights[i].Elements[j] = Parent2.RecurrentWeights[i].Elements[j];
        }
        
        //randomly mutate genes
        for (int k = 0; k < offspringnetwork.Biases[i].Elements.size(); k++)
        {
            int random_int = rand() % (int)((1.01f - mutation_probability) * 1000); //we round mutation_probability * 1000 to an integer to ensure it is not a double
            for (int j = 0; j < 10; j++) //we're choosing out of 1000 to get precision up to the hundredth place, so we must take 10 samples to get a probability out of 100
            {
                //random selection of gene
                if (random_int == rand() % (int)(mutation_probability * 1000))
                {
                    offspringnetwork.Biases[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
                    offspringnetwork.RecurrentWeights[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
                }
            }
        }
    }
    
    return offspringnetwork;
}

std::vector<std::vector<Matrix> > RNN::FeedForward(Matrix input, int num_passes)
{
    std::vector<std::vector<Matrix> > networkdata;
    Matrix prev_output;
    for (int i = 0; i < num_passes; i++)
    {
        if (i == 0)
        {
            std::vector<Matrix> zerorecurrence;
            for (int j = 0; j < Num_Layers; j++)
            {
                Matrix zr(InputVectorSize, 1);
                zerorecurrence.push_back(zr);
            }
            networkdata.push_back(PartialFeedFoward(input, zerorecurrence));
        }
        else
        {
            Matrix m(networkdata[networkdata.size() - 1][Num_Layers - 1].Elements.size(), 1);
            m.Elements[MaxElement(networkdata[networkdata.size() - 1][Num_Layers - 1])] = 1.0f;
            networkdata.push_back(PartialFeedFoward(m, networkdata[networkdata.size() - 1]));
        }
    }
    
    return networkdata;
}

std::vector<Matrix> RNN::PartialFeedFoward(Matrix input, std::vector<Matrix> recurrences)
{
    std::vector<Matrix> outputs;
    //Add biases and apply activation function to each input element
    for (int i = 0; i < input.Dimensions[0]; i++)
    {
        input.Elements[i] += Biases[0].Elements[i];
        input.Elements[i] = 1 / (1 + pow(2.718281828459f, -input.Elements[i]));
    }
    outputs.push_back(input);
    
    //feed forward calculation
    for (int i = 1; i < Num_Layers; i++)
    {
        //feed forward
        Matrix z;
        z = Weights[i].Transpose() * outputs[i - 1] + Biases[i];
        
        if (i != Num_Layers - 1)
            for (int j = 0; j < z.Dimensions[0]; j++)
                z.Elements[j] += recurrences[i].Elements[j] * RecurrentWeights[i].Elements[j];
        
        outputs.push_back(z);
        
        //Apply activation function
        for (int j = 0; j < outputs[i].Dimensions[0]; j++)
        {
            outputs[i].Elements[j] = 1 / (1 + pow(2.718281828459f, -outputs[i].Elements[j]));
        }
    }
    
    return outputs;
}

double RNN::TrainWithBackPropagation(std::vector<Matrix> sequence, double learning_rate)
{
    double cost = 0.0f;
    
    BackpropagationData data1; //data from first step
    BackpropagationData data2; //data from second step
    
    for (int i = 0; i < sequence.size() - 2; i++)
    {
        if (i == 0)
        {
            std::vector<Matrix> zerorecurrence;
            for (int j = 0; j < Num_Layers; j++)
            {
                Matrix zr(InputVectorSize, 1);
                zerorecurrence.push_back(zr);
            }
            data1 = TrainStep(sequence[i], sequence[i + 1], zerorecurrence);
            data2 = TrainStep(sequence[i + 1], sequence[i + 2], data1.activations);
        }
        else
        {
            data1 = TrainStep(sequence[i], sequence[i + 1], data2.activations);
            data2 = TrainStep(sequence[i + 1], sequence[i + 2], data1.activations);
        }
        
        //update biases
        for (int i = 0; i < Biases.size(); i++)
        {
            Biases[i] = Biases[i] + data1.deltas[i] * (-1.0f * learning_rate);
        }
        
        //update weights
        for (int i = 1; i < Weights.size(); i++)
        {
            Weights[i] = Weights[i] + ((data1.activations[i - 1] * data1.deltas[i].Transpose()) * (-1.0f * learning_rate));
        }
        
        //update biases
        for (int i = 0; i < Biases.size(); i++)
        {
            Biases[i] = Biases[i] + data2.deltas[i] * (-1.0f * learning_rate);
        }
        
        //update weights
        for (int i = 1; i < Weights.size(); i++)
        {
            Weights[i] = Weights[i] + ((data2.activations[i - 1] * data2.deltas[i].Transpose()) * (-1.0f * learning_rate));
        }
        
        //update recurrent weights
        for (int i = 1; i < RecurrentWeights.size() - 1; i++)
        {
            for (int j = 0; j < RecurrentWeights[i].Elements.size(); j++)
                RecurrentWeights[i].Elements[j] -= data1.activations[i].Elements[j] * data2.deltas[i + 1].Elements[j] * learning_rate;
        }
        
        cost += data1.cost + data2.cost;
    }
    
    return cost;
}

BackpropagationData RNN::TrainStep(Matrix input, Matrix output, std::vector<Matrix> recurrences)
{
    BackpropagationData data;
    std::vector<Matrix> outputs = PartialFeedFoward(input, recurrences);
    
    std::vector<Matrix> temp_deltas; //layer deltas stored backwards in order
    
    //calculate cost function
    double cost = 0.0f;
    Matrix partial_cost_matrix(InputVectorSize, 1);
    partial_cost_matrix = output + (outputs[outputs.size() - 1] * -1);
    for (int i = 0; i < partial_cost_matrix.Elements.size(); i++)
    {
        cost += 0.5f * partial_cost_matrix.Elements[i] * partial_cost_matrix.Elements[i];
    }
    //calculate last layer deltas
    Matrix lld(InputVectorSize, 1);
    lld = outputs[outputs.size() - 1] + (output * -1);
    for (int i = 0; i < lld.Dimensions[0]; i++)
    {
        double a = outputs[outputs.size() - 1].Elements[i];
        lld.Elements[i] *= a * (1 - a); //derivative of activation function
    }
    temp_deltas.push_back(lld);
    
    //calculate the rest of the deltas through back propagation
    int j = 0; //this keeps track of the index for the next layer's delta
    for (int i = Num_Layers - 2; i >= 0; i--) //start at the second to last layer
    {
        Matrix delta(InputVectorSize, 1);
        delta = Weights[i + 1] * temp_deltas[j];
        j++;
        for (int k = 0; k < delta.Dimensions[0]; k++)
        {
            double a = outputs[i].Elements[k];
            delta.Elements[k] *= a * (1 - a); //derivative of activation function
        }
        temp_deltas.push_back(delta);
    }
    
    //put the deltas into a new vector object in the correct order
    std::vector<Matrix> deltas;
    for (int i = (int)temp_deltas.size() - 1; i >= 0; i--)
    {
        deltas.push_back(temp_deltas[i]);
    }
    
    data.cost = cost;
    data.activations = outputs;
    data.deltas = deltas;
    
    return data;
}
*/

//This convolutional neural network code is not implemented correctly and does not function correctly
/*
CovNet28x28::CovNet28x28()
{
    Matrix l1b(28, 28);
    l1b.Randomize();
    Layer1_Bias = l1b;
    
    for (int i = 0; i < 9; i++)
    {
        Matrix k5x5(5, 5);
        k5x5.Randomize();
        Kernels1.push_back(k5x5);
    }
    
    for (int i = 0; i < 9; i++)
    {
        Matrix l3b(12, 12);
        l3b.Randomize();
        Layer3_Biases.push_back(l3b);
    }
    
    for (int i = 0; i < 9; i++)
    {
        Matrix k3x3(3, 3);
        k3x3.Randomize();
        Kernels2.push_back(k3x3);
    }
    
    int dimensions[2] = {900, 10};
    FFANN fullyconnectedlayer(dimensions, 2);
    FullyConnectedLayer = fullyconnectedlayer;
}

//COVNET IS NOT WORKING WELL YET
std::vector<std::vector<Matrix> > CovNet28x28::FeedForward(Matrix input)
{
    std::vector<std::vector<Matrix> > outputs;
    //calculate first layer activations
    input = input + Layer1_Bias;
    for (int i = 0; i < input.Elements.size(); i++)
        input.Elements[i] = 1 / (1 + pow(2.718281828459f, -input.Elements[i]));
    
    std::vector<Matrix> input_vector;
    input_vector.push_back(input);
    
    outputs.push_back(input_vector);
    
    std::vector<Matrix> Layers2;
    
    for (int a = 0; a < 9; a++)
    {
        Matrix Layer2(24, 24);
        //first convolution
        for (int i = 0; i < 24; i++)
            for (int j = 0; j < 24; j++)
                for (int k = 0; k < 5; k++)
                    for (int l = 0; l < 5; l++)
                        Layer2.Elements[CoordinateToIndex(j, i, &Layer2)] += input.Elements[CoordinateToIndex(j + l, i + k, &input)]
                        * Kernels1[a].Elements[CoordinateToIndex(l, k, &Kernels1[a])];
        Layers2.push_back(Layer2);
    }
    
    outputs.push_back(Layers2);
    
    std::vector<Matrix> Layers3;
    
    for (int a = 0; a < 9; a++)
    {
        //Max pooling
        Matrix Layer3(12, 12);
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++)
            {
                Matrix m(2, 2);
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                        m.Elements[CoordinateToIndex(l, k, &m)] = Layers2[a].Elements[CoordinateToIndex(j * 2 + l, i * 2 + k, &Layers2[a])];
                Layer3.Elements[CoordinateToIndex(j, i, &Layer3)] = m.Elements[MaxElement(m)];
            }
        Layers3.push_back(Layer3);
    }
    
    for (int j = 0; j < 9; j++)
    {
        //calculate third layer activations
        Layers3[j] = Layers3[j] + Layer3_Biases[j];
        for (int i = 0; i < Layers3[j].Elements.size(); i++)
            Layers3[j].Elements[i] = 1 / (1 + pow(2.718281828459f, -Layers3[j].Elements[i]));
    }
    
    outputs.push_back(Layers3);
    
    std::vector<Matrix> Layers4;
    
    for (int a = 0; a < 9; a++)
    {
        Matrix Layer4(10, 10);
        //second convolution
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                        Layer4.Elements[CoordinateToIndex(j, i, &Layer4)] += Layers3[a].Elements[CoordinateToIndex(j + l, i + k, &Layers3[a])]
                        * Kernels2[a].Elements[CoordinateToIndex(l, k, &Kernels2[a])];
        Layers4.push_back(Layer4);
    }
    
    //full feed forward
    Matrix fclinput(900, 1);
    for (int j = 0; j < Layers4.size(); j++)
    {
        for (int i = 0; i < 100; i++)
        {
            fclinput.Elements[j * 100 + i] = Layers4[j].Elements[i];
        }
    }
    
    std::vector<Matrix> fcloutput = FullyConnectedLayer.FeedForward(fclinput);
    
    for (int i = 0; i < fcloutput.size(); i++)
    {
        std::vector<Matrix> fcloutput_vector;
        fcloutput_vector.push_back(fcloutput[i]);
        outputs.push_back(fcloutput_vector);
    }
    
    return outputs;
}

double CovNet28x28::TrainWithBackPropagation(Matrix input, Matrix output, double learning_rate)
{
    std::vector<std::vector<Matrix> > outputs = FeedForward(input);
    //train the fully connected layer and store the deltas of the first layer
    Matrix lld(900, 1);
    std::vector<Matrix> outputs_l2l; //outputs of the feedfoward network
    outputs_l2l.push_back(outputs[outputs.size() - 2][0]);
    outputs_l2l.push_back(outputs[outputs.size() - 1][0]);
    
    double cost = FullyConnectedLayer.TrainWithBackPropagation(input, output, outputs_l2l, learning_rate, &lld);
    
    std::vector<Matrix> llds;
    
    for (int i = 0; i < 9; i++)
    {
        Matrix templld(10, 10);
        for (int j = 0; j < 100; j++)
        {
            templld.Elements[j] = lld.Elements[i * 100 + j];
        }
        llds.push_back(templld);
    }
    
    //calculate Layer 3 deltas
    std::vector<Matrix> layers3_deltas;
    for (int b = 0; b < 9; b++)
    {
        Matrix layer3_deltas(12, 12);
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                    {
                        double a = outputs[2][b].Elements[CoordinateToIndex(j + l, i + k, &outputs[2][b])];
                        double derivative = a * (1 - a);
                        layer3_deltas.Elements[CoordinateToIndex(j + l, i + k, &layer3_deltas)] += llds[b].Elements[CoordinateToIndex(j, i, &llds[b])] * derivative;
                    }
        layers3_deltas.push_back(layer3_deltas);
    }
    
    //calculate Layer 2 deltas
    std::vector<Matrix> layers2_deltas;
    for (int a = 0; a < 9; a++)
    {
        Matrix layer2_deltas(24, 24);
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++)
            {
                Matrix m(2, 2);
                int x;
                int y;
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                        m.Elements[CoordinateToIndex(l, k, &m)] = outputs[1][a].Elements[CoordinateToIndex(j * 2 + l, i * 2 + k, &outputs[1][a])];
                x = j * 2 + MaxElement(m) % 2;
                y = i * 2 + (int)(MaxElement(m) / 2);
                layer2_deltas.Elements[CoordinateToIndex(x, y, &layer2_deltas)] = layers3_deltas[a].Elements[CoordinateToIndex(j, i, &layers3_deltas[a])];
            }
        layers2_deltas.push_back(layer2_deltas);
    }
    
    //calculate Layer 1 deltas
    Matrix layer1_deltas(28, 28);
    for (int b = 0; b < 9; b++)
    {
        for (int i = 0; i < 24; i++)
            for (int j = 0; j < 24; j++)
                for (int k = 0; k < 5; k++)
                    for (int l = 0; l < 5; l++)
                    {
                        double a = outputs[0][0].Elements[CoordinateToIndex(j + l, i + k, &outputs[0][0])];
                        double derivative = a * (1 - a);
                        layer1_deltas.Elements[CoordinateToIndex(j + l, i + k, &layer1_deltas)] += layers2_deltas[b].Elements[CoordinateToIndex(j, i, &layers2_deltas[b])] * derivative;
                    }
    }
    
    //update biases
    for (int i = 0; i < Layer1_Bias.Elements.size(); i++)
        Layer1_Bias.Elements[i] -= layer1_deltas.Elements[i] * learning_rate;
    
    for (int a = 0; a < 9; a++)
        for (int i = 0; i < Layer3_Biases[a].Elements.size(); i++)
            Layer3_Biases[a].Elements[i] -= layers3_deltas[a].Elements[i] * learning_rate;
    
    //update kernels
    for (int a = 0; a < 9; a++)
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                        Kernels2[a].Elements[CoordinateToIndex(l, k, &Kernels2[a])] -= outputs[2][a].Elements[CoordinateToIndex(j + l, i + k, &outputs[2][a])] * lld.Elements[CoordinateToIndex(j, i, &lld)] * learning_rate;
    
    for (int a = 0; a < 9; a++)
        for (int i = 0; i < 24; i++)
            for (int j = 0; j < 24; j++)
                for (int k = 0; k < 5; k++)
                    for (int l = 0; l < 5; l++)
                        Kernels1[a].Elements[CoordinateToIndex(l, k, &Kernels1[a])] -= outputs[0][0].Elements[CoordinateToIndex(j + l, i + k, &outputs[0][0])] * layers2_deltas[a].Elements[CoordinateToIndex(j, i, &layers2_deltas[a])] * learning_rate;
    
    return cost;
}
*/

int MaxElement(Matrix m)
{
    if (m.Elements.size() == 0)
        return 0;
    double max = m.Elements[0];
    int max_i = 0;
    for (int i = 0; i < m.Elements.size(); i++)
    {
        if (m.Elements[i] > max)
        {
            max = m.Elements[i];
            max_i = i;
        }
    }
    return max_i;
}
