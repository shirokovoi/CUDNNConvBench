//
// Created by oleg on 23.07.17.
//

#ifndef CUDNNCONVBENCH_TYPES_HPP
#define CUDNNCONVBENCH_TYPES_HPP

#include <cudnn.h>
#include <string>
#include <map>

struct Size
{
	Size();
	Size(int width, int height);

	std::string ToString();
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

	std::map<cudnnConvolutionFwdAlgo_t, int> m_ForwardTimes;
	std::map<cudnnConvolutionBwdDataAlgo_t, int> m_BackwardDataTimes;
	std::map<cudnnConvolutionBwdFilterAlgo_t, int> m_BackwardFilterTimes;
};

std::string FwdAlgo_ToString(cudnnConvolutionFwdAlgo_t algo);
std::string BwdData_ToString(cudnnConvolutionBwdDataAlgo_t algo);
std::string BwdFilter_ToString(cudnnConvolutionBwdFilterAlgo_t algo);

#endif //CUDNNCONVBENCH_TYPES_HPP
