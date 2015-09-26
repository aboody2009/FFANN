#pragma once
#ifndef MATRIXMATH_H
#define MATRIXMATH_H

#include <iostream>
#include <cstdlib>
#include <vector>

class Matrix
{
public:
	Matrix();
	~Matrix();
	Matrix(int row_dim, int col_dim);
	Matrix(int row_dim, int col_dim, double* elements);
	Matrix operator*(const Matrix& m);
	Matrix operator*(const double& f);
	Matrix operator+(const Matrix& m);
	Matrix Transpose();
	void CoutMatrix();
	int Dimensions[2];
	std::vector<double> Elements;
};

#endif
