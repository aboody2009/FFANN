//Sully Chen 2015
//This is example code which creates a neural network and trains it to recognize the larger number out of a pair of doubles ranging -1 to 1
//I ran this code with 1000000 examples at a learning rate of 0.001, and achieved 95% to 99% accuracy. I recommend these training settings
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <time.h>
#include <fstream>
#include "MatrixMath.h"
#include "FFANN.h"

int main()
{
	//seed random number generator
	srand((unsigned int)time(NULL));
	//create structure of neural network: 2 input neurons, 2 output neurons
	int dimensions[3] = { 2, 3, 2 };
	//create the neural network
	FFANN testFFANN(dimensions, 3);

	int num_examples;
	double learning_rate = 0.0f;

	std::cout << "Train how many trials?: ";
	std::cin >> num_examples;
	std::cout << "At what learning rate?: ";
	std::cin >> learning_rate;

	const double original_learning_rate = learning_rate;

	std::ofstream saveFile("cost_function_data.txt");

	//run training
	for (int i = 0; i < num_examples; i++)
	{
		//create inputs
		double inputs[2] = { (rand() % 200 - 100) / 100.0f, (rand() % 200 - 100) / 100.0f };
		if (inputs[0] == inputs[1])
			inputs[0] += (rand() % 200 - 100) / 1000.0f;

		//create a 2x1 matrix
		Matrix input(2, 1, inputs);

		//create outputs based on which input is larger
		double outputs[2];
		if (inputs[0] > inputs[1])
		{
			outputs[0] = 1.0f;
			outputs[1] = 0.0f;
		}
		else
		{
			outputs[0] = 0.0f;
			outputs[1] = 1.0f;
		}
		Matrix output(2, 1, outputs);

		//lower learning rate near the end of training
		if (i > num_examples * 0.8f)
			learning_rate = original_learning_rate * 0.1f;

		//save cost function data
		double cost = testFFANN.TrainWithBackPropagation(input, output, learning_rate);
		if (i % (int)(num_examples / 100.0f) == 0)
		{
			saveFile << i << ", " << cost;
		}
		if (i % (int)(num_examples / 100.0f) == 0)
		{
			saveFile << "\n";
		}
	}
	saveFile.close();
	std::cout << "Done training!\n" << std::endl << "Testing feed forward neural network...\n" << std::endl;

	//test the network
	int num_trials = 10000;
	int num_correct = 0;

	for (int i = 0; i < num_trials; i++)
	{
		double inputs[2] = { 0.0f, 0.0f };
		while (inputs[0] == inputs[1])
		{
			inputs[0] = (rand() % 200 - 100) / 100.0f;
			inputs[1] = (rand() % 200 - 100) / 100.0f;
		}
		Matrix input(2, 1, inputs);
		std::vector<Matrix> outputs = testFFANN.FeedForward(input);
		if (inputs[0] > inputs[1])
		{
			if (outputs[outputs.size() - 1].Elements[0] > outputs[outputs.size() - 1].Elements[1])
				num_correct++;
		}
		else
		{
			if (outputs[outputs.size() - 1].Elements[0] < outputs[outputs.size() - 1].Elements[1])
				num_correct++;
		}
	}

	std::cout << "Done Testing! Here are the results:\n" << std::endl << "Accuracy: " << num_correct * 100 / num_trials << "% correct" << std::endl;

	system("PAUSE");

	std::cout << "Continue with sine wave function approximation test? y/n: ";
	char yesorno;
	std::cin >> yesorno;
	if (yesorno != 'y')
		return 0;
	std::cout << "\n";

	int dimensions2[3] = { 1, 20, 1 };

	std::ofstream saveFile1("accuracy_cost_data_for_sine_wave.txt");

	FFANN SineFFANN(dimensions2, 3);

	double increment = 0.1f;

	for (int i = 0; i < 1000000; i++)
	{
		double cost = 0.0f;
		for (double j = -1.0f; j <= 1.0f; j += increment)
		{
			Matrix m1(1, 1);
			m1.Elements[0] = j;
			Matrix m2(1, 1);
			m2.Elements[0] = 0.5f * sin(3.1415926535f * j) + 0.5f;
			if (i < 100000)
			{
				cost += SineFFANN.TrainWithBackPropagation(m1, m2, 0.1f);
			}
			else if (i < 500000)
			{
				cost += SineFFANN.TrainWithBackPropagation(m1, m2, 0.01f);
			}
			else if (i < 900000)
			{
				cost += SineFFANN.TrainWithBackPropagation(m1, m2, 0.001f);
			}
			else
			{
				cost += SineFFANN.TrainWithBackPropagation(m1, m2, 0.0001f);
            }
		}

		if (i % 1000 == 0)
		{
			std::cout << "iteration: " << i << std::endl;
			std::cout << "cost function: " << cost / (2.0f / increment) << std::endl;
			double average_error = 0.0f;
			for (double j = -1.0f; j <= 1.0f; j += increment)
			{
				Matrix m1(1, 1);
				m1.Elements[0] = j;
				Matrix m2(1, 1);
				m2.Elements[0] = 0.5f * sin(3.1415926535f * j) + 0.5f;
				std::vector<Matrix> mvec = SineFFANN.FeedForward(m1);
				average_error += std::abs(mvec[mvec.size() - 1].Elements[0] - m2.Elements[0]);
			}
			std::cout << "accuracy: " << (1 - average_error / (2.0f / increment)) * 100 << "%\n" << std::endl;
			saveFile1 << i << ", " << 1 - average_error / (2.0f / increment) << ", " << cost / (2.0f / increment) << std::endl;
		}
	}

	std::ofstream saveFile2("after_training_output.txt");

	for (double j = -1.0f; j <= 1.0f; j += 0.01f)
	{
		Matrix m1(1, 1);
		m1.Elements[0] = j;
		std::vector<Matrix> mvec = SineFFANN.FeedForward(m1);
		saveFile2 << j << ", " << mvec[mvec.size() - 1].Elements[0] << std::endl;
	}

	saveFile2.close();

	std::cout << "\nType \"c\" to continue to RNN test, otherwise type something else and the program will quit" << std::endl;
	char answer;
	std::cin >> answer;
	if (answer != 'c')
		return 0;
	std::cout << "\n\n\nRunning RNN Test..." << std::endl;

	/*
	//create a recurrent neural network with an input vector size of 5 and with 3 layers
	RNN testRNN(5 , 3);
	std::vector<std::vector<Matrix> > rnnoutputdata;
	Matrix rnninput(5, 1);
	for (int i = 0; i < 5; i++)
	{
	rnninput.Elements[i] = (rand() % 200 - 100) / 100.0f;
	}
	rnnoutputdata = testRNN.FeedForward(rnninput, 10);
	for (int i = 0; i < rnnoutputdata.size(); i++)
	{
	for (int j = 0; j < rnnoutputdata[i].size(); j++)
	{
	rnnoutputdata[i][j].CoutMatrix();
	std::cout << "\n";
	}
	std::cout << "\n";
	std::cout << "\n";
	}
	*/


	//RNN CODE NOT WORKING YET
	RNN testRNN2(10, 3);

	//generate sequence
	std::vector<Matrix> sequence;
	for (int i = 0; i < 10; i++)
	{
		Matrix s(10, 1);
		s.Elements[i] = 1.0f;
		sequence.push_back(s);
	}

	for (int i = 0; i < 1000000; i++)
	{
		double learning_rate = 0.001f;
		if (i > 800000)
			learning_rate = 0.0001f;
		if (i > 950000)
			learning_rate = 0.00001f;
		testRNN2.TrainWithBackPropagation(sequence, learning_rate);
		if (i % 1000 == 0)
			std::cout << i << " iterations complete" << std::endl;
	}

	//test the network
	std::vector<std::vector<Matrix> > RNNTestData = testRNN2.FeedForward(sequence[0], 10);
	for (int i = 0; i < RNNTestData.size(); i++)
	{
		double max = 0.0f;
		int max_j = 0;
		for (int j = 0; j < testRNN2.InputVectorSize; j++)
		{
			if (RNNTestData[i][RNNTestData[i].size() - 1].Elements[j] > max)
			{
				max = RNNTestData[i][RNNTestData[i].size() - 1].Elements[j];
				max_j = j;
			}
		}
		std::cout << max_j << std::endl;
	}

	system("PAUSE");

	return 0;
}
