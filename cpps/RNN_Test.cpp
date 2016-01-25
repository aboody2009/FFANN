//Sully Chen 2015
//This example trains a recurrent neural network to count from 0 to 10

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include "MatrixMath.h"
#include "FFANN.h"

int main()
{
    //seed random number generator
    srand((unsigned int)time(NULL));
    RNN testRNN2(10, 3);
    
    //generate sequence
    std::vector<Matrix> sequence;
    for (int i = 0; i < 10; i++)
    {
        Matrix s(10, 1);
        s.Elements[i] = 1.0f;
        sequence.push_back(s);
    }
    
    for (int i = 0; i < 10000; i++)
    {
        double learning_rate = 0.1f;
        double cost = testRNN2.TrainWithBackPropagation(sequence, learning_rate);
        if (i % 100 == 0)
            std::cout << i << " iterations complete" << ", cost: " << cost << std::endl;
    }
    
    //test the network
    std::vector<std::vector<Matrix> > RNNTestData = testRNN2.FeedForward(sequence[0], 10);
    for (int i = 0; i < RNNTestData.size(); i++)
        std::cout << MaxElement(RNNTestData[i][RNNTestData[i].size() - 1]) << std::endl;
    
    system("PAUSE");
    
    return 0;
}
