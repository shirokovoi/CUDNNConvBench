//
// Created by oleg on 23.07.17.
//

#ifndef CUDNNCONVBENCH_TYPES_HPP
#define CUDNNCONVBENCH_TYPES_HPP

#include <cudnn.h>
#include <string>

struct Size
{
	Size();

	int m_Width;
	int m_Height;
};

struct Problem
{
	Problem();

	bool Parse(const std::string& line);
	std::string ToString();
	Size m_InputSize;
	Size m_FilterSize;
	Size m_Stride;
	Size m_Padding;

	int m_BatchSize;
	int m_InputMaps;
	int m_OutputMaps;
};

struct Result
{
	Result();

	std::string ToString();
	Problem m_Problem;
	uint32_t m_Repeats;
	int m_ForwardElapsedUSec;
	int m_BackwardFilterElapsedUSec;
	int m_BackwardDataElapsedUSec;
};

std::ostream& operator<<(std::ostream& s, cudnnConvolutionFwdAlgo_t algo);
std::ostream& operator<<(std::ostream& s, cudnnConvolutionBwdDataAlgo_t algo);
std::ostream& operator<<(std::ostream& s, cudnnConvolutionBwdFilterAlgo_t algo);

#endif //CUDNNCONVBENCH_TYPES_HPP
