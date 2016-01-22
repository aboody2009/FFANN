//Sully Chen
//01/22/2016

#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <SFML/Graphics.hpp>
#include "FFANN.h"
#include "MatrixMath.h"

int main(int, char const**)
{
    srand(time(NULL));
    //open the MNIST dataset
    std::ifstream data0;
    data0.open("MNIST_Dataset/data0.txt", std::ios::binary);
    std::ifstream data1;
    data1.open("MNIST_Dataset/data1.txt", std::ios::binary);
    std::ifstream data2;
    data2.open("MNIST_dataset/data2.txt", std::ios::binary);
    std::ifstream data3;
    data3.open("MNIST_Dataset/data3.txt", std::ios::binary);
    std::ifstream data4;
    data4.open("MNIST_Dataset/data4.txt", std::ios::binary);
    std::ifstream data5;
    data5.open("MNIST_Dataset/data5.txt", std::ios::binary);
    std::ifstream data6;
    data6.open("MNIST_Dataset/data6.txt", std::ios::binary);
    std::ifstream data7;
    data7.open("MNIST_Dataset/data7.txt", std::ios::binary);
    std::ifstream data8;
    data8.open("MNIST_Dataset/data8.txt", std::ios::binary);
    std::ifstream data9;
    data9.open("MNIST_Dataset/data9.txt", std::ios::binary);
    
    std::vector<Matrix> input0;
    std::vector<Matrix> input1;
    std::vector<Matrix> input2;
    std::vector<Matrix> input3;
    std::vector<Matrix> input4;
    std::vector<Matrix> input5;
    std::vector<Matrix> input6;
    std::vector<Matrix> input7;
    std::vector<Matrix> input8;
    std::vector<Matrix> input9;
    
    //create the input vectors
    while (data0.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data0.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input0.push_back(m);
    }
    while (data1.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data1.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input1.push_back(m);
    }
    while (data2.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data2.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input2.push_back(m);
    }
    while (data3.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data3.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input3.push_back(m);
    }
    while (data4.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data4.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input4.push_back(m);
    }
    while (data5.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data5.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input5.push_back(m);
    }
    while (data6.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data6.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input6.push_back(m);
    }
    while (data7.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data7.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input7.push_back(m);
    }
    while (data8.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data8.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input8.push_back(m);
    }
    while (data9.good())
    {
        Matrix m(784, 1);
        for (int i = 0; i < 784; i++)
        {
            int data_byte = data9.get();
            m.Elements[i] = data_byte / 255.0f;
        }
        input9.push_back(m);
    }
    
    //create the labels
    std::vector<Matrix> labels;
    for (int i = 0; i < 10; i++)
    {
        Matrix m(10, 1);
        m.Elements[i] = 1.0f;
        labels.push_back(m);
    }
    
    data0.close();
    data1.close();
    data2.close();
    data3.close();
    data4.close();
    data5.close();
    data6.close();
    data7.close();
    data8.close();
    data9.close();
    
    int dimensions[] = {784, 10};
    
    FFANN MNISTModel(dimensions, 2);
    
    //train the model using backpropagation
    for (int j = 0; j < 10; j++)
    {
        for (int i = 0; i < input0.size() - 100; i++) //all input vectors have the same number of images, so we just iterate using the size of input0, and use the last 100 images from each dataset for testing
        {
            double cost = 0.0f;
            double learning_rate = 0.1f / pow(10, j / 2.0f); //slowly lower the learning rate
            cost += MNISTModel.TrainWithBackPropagation(input0[i], labels[0], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input1[i], labels[1], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input2[i], labels[2], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input3[i], labels[3], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input4[i], labels[4], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input5[i], labels[5], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input6[i], labels[6], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input7[i], labels[7], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input8[i], labels[8], learning_rate);
            cost += MNISTModel.TrainWithBackPropagation(input9[i], labels[9], learning_rate);
            cost /= 10.0f;
            std::cout << "Iteration: " << i + j * 900 << " complete, " << "cost: " << cost << std::endl;
        }
    }
    
    std::cout << "Testing model..." << std::endl;
    
    int num_correct = 0;
    int num_trials = 0;
    
    for (int i = input0.size() - 100; i < input0.size(); i++)
    {
        std::vector<Matrix> output = MNISTModel.FeedForward(input0[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[0]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input1[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[1]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input2[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[2]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input3[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[3]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input4[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[4]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input5[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[5]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input6[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[6]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input7[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[7]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input8[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[8]))
            num_correct++;
        
        output = MNISTModel.FeedForward(input9[i]); //feedforward the input vector
        if (MaxElement(output[output.size() - 1]) == MaxElement(labels[9]))
            num_correct++;
        
        //output[output.size() - 1].CoutMatrix();
        
        num_trials += 10;
        
        std::cout << "Testing model... iteration: " << i - input0.size() + 100 << std::endl;
    }
    
    double accuracy = (double)num_correct / (double)num_trials;
    
    std::cout << "Accuracy: " << accuracy * 100 << "% correct" << std::endl;
    // Create the main window
    sf::RenderWindow window(sf::VideoMode(280, 280), "Handwriting Recognition");
    sf::VertexArray pointmap(sf::Points, 280 * 280);
    for (int i = 0; i < 280; i++)
        for (int j = 0; j < 280; j++)
        {
            pointmap[i * 280 + j].position.x = j;
            pointmap[i * 280 + j].position.y = i;
            pointmap[i * 280 + j].color = sf::Color::Black;
        }
    
    // Start the game loop
    while (window.isOpen())
    {
        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window: exit
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // Escape pressed: exit
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
                window.close();
            }
        }
        
        //zoom into area that is left clicked
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        {
            sf::Vector2i position = sf::Mouse::getPosition(window);
            if (position.x + position.y * 280 < 280*280 && position.x + position.y * 280 >= 0)
            {
                for (int i = -16; i < 17; i++)
                {
                    for (int j = -16; j < 17; j++)
                    {
                        if (position.x + i + (position.y + j) * 280 < 280*280 && position.x + i + (position.y + j) * 280 >= 0)
                        {
                            double distance_squared = i * i + j * j + 1;
                            sf::Color color(255 / distance_squared, 255 / distance_squared, 255 / distance_squared);
                            pointmap[position.x + i + (position.y + j) * 280].position.x = position.x + i;
                            pointmap[position.x + i + (position.y + j) * 280].position.y = position.y + j;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                        }
                            
                    }
                }
            }
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C))
        {
            for (int i = 0; i < 280*280; i++)
                pointmap[i].color = sf::Color::Black;
        }
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Return))
        {
            Matrix input(784, 1);
            for (int k = 0; k < 28; k++)
            {
                for (int l = 0; l < 28; l++)
                {
                    double average = 0.0f;
                    for (int i = 0; i < 10; i++)
                        for (int j = 0; j < 10; j++)
                        {
                            double temp_average = 0.0f;
                            temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.r;
                            temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.g;
                            temp_average += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.b;
                            temp_average /= 3.0f;
                            average += temp_average;
                        }
                    average /= 100.0f;
                    average /= 255.0f; //normalize
                    input.Elements[k * 28 + l] = average;
                }
            }
            std::vector<Matrix> output = MNISTModel.FeedForward(input);
            output[output.size() - 1].CoutMatrix();
            int prediction = MaxElement(output[output.size() - 1]);
            std::cout << "This number is predicted to be a: " << prediction << std::endl;
        }
        
        // Clear screen
        window.clear();
        window.draw(pointmap);
        // Update the window
        window.display();
    }

    return EXIT_SUCCESS;
}

