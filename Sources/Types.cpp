//
// Created by oleg on 23.07.17.
//

#include <Types.hpp>
#include <sstream>
#include <vector>
#include <algorithm>

Size::Size():
    m_Width(0),
    m_Height(0)
{ }

std::string Size::ToString()
{
	std::stringstream ss;
	ss << m_Width << ", " << m_Height;
	return ss.str();
}

Problem::Problem():
    m_InputMaps(0),
    m_OutputMaps(0)
{ }

template<typename T>
bool TryParse(const std::string& str, T& out)
{
	std::stringstream ss;
	T tmp = T();
	ss << str;
	ss >> tmp;
	if (ss.fail())
		return false;

	out = tmp;
	return true;
}

std::vector<std::string> Split(const std::string& str, const std::string& delimiter)
{
	std::vector<std::string> result;
	size_t old_pos = 0;
	size_t pos;

	while ((pos = str.find(delimiter, old_pos)) != std::string::npos)
	{
		result.emplace_back(str.substr(old_pos, pos - old_pos));
		old_pos = pos + delimiter.size();
	}
	result.emplace_back(str.substr(old_pos));

	return result;
}

std::string Trim(const std::string& str)
{
	std::string result = str;
	result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](int ch){ return !std::isspace(ch); }));
	result.erase(std::find_if(result.rbegin(), result.rend(), [](int ch){ return !std::isspace(ch); }).base(), result.end());
	return result;
}

bool Problem::Parse(const std::string& line)
{
	auto tokens_strs = Split(line, ",");
	std::vector<int> tokens;
	std::transform(tokens_strs.begin(), tokens_strs.end(), tokens.begin(),
	               [](const std::string& str)
	               {
		               int out;
		               TryParse(Trim(str), out);
		               return out;
	               });

	m_InputSize.m_Width = tokens[0];
	m_InputSize.m_Height = tokens[1];
	m_FilterSize.m_Width = tokens[2];
	m_FilterSize.m_Height = tokens[3];
	m_Stride.m_Width = tokens[4];
	m_Stride.m_Height = tokens[5];
	m_Padding.m_Width = tokens[6];
	m_Padding.m_Height = tokens[7];
	m_BatchSize = tokens[8];
	m_InputMaps = tokens[9];
	m_OutputMaps = tokens[10];

	return true;
}

std::string Problem::ToString()
{
	std::stringstream ss;
	ss << m_InputSize.ToString() << ", "
       << m_FilterSize.ToString() << ", "
       << m_Stride.ToString() << ", "
       << m_Padding.ToString() << ", "
       << m_BatchSize << ", "
       << m_InputMaps << ", "
       << m_OutputMaps;

	return ss.str();
}

Result::Result():
    m_Repeats(0),
    m_ForwardElapsedUSec(0),
    m_BackwardFilterElapsedUSec(0),
    m_BackwardDataElapsedUSec(0)
{ }

std::string Result::ToString()
{
	std::stringstream ss;
	ss << m_Problem.ToString() << ", " << m_Repeats << ", " << m_ForwardElapsedUSec << ", "
       << FwdAlgo_ToString(m_ForwardAlgo) << ", " << BwdFilter_ToString(m_BackwardFilterAlgo) << ", " << BwdData_ToString(m_BackwardDataAlgo) << ", "
       << m_ForwardElapsedUSec << ", " << m_BackwardFilterElapsedUSec << ", " << m_BackwardDataElapsedUSec;

	return ss.str();
}

std::string FwdAlgo_ToString(cudnnConvolutionFwdAlgo_t algo)
{
	std::stringstream ss;
	switch (algo)
	{
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
			ss << "Implicit GEMM";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
			ss << "Implicit Precomp GEMM";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
			ss << "GEMM";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
			ss << "DIRECT";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
			ss << "FFT";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
			ss << "FFT_TILING";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
			ss << "WINOGRAD";
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
			ss << "WINOGRAD_NONFUSED";
			break;
	}

	return ss.str();
}

std::string BwdData_ToString(cudnnConvolutionBwdDataAlgo_t algo)
{
	std::stringstream ss;
	switch (algo)
	{
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
			ss << "ALGO_0";
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
			ss << "ALGO_1";
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
			ss << "FFT";
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
			ss << "FFT_TILING";
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
			ss << "WINOGRAD";
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
			ss << "WINOGRAD_NONFUSED";
			break;
	}

	return ss.str();
}

std::string BwdFilter_ToString(cudnnConvolutionBwdFilterAlgo_t algo)
{
	std::stringstream ss;
	switch (algo)
	{
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
			ss << "ALGO_0";
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
			ss << "ALGO_1";
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
			ss << "FFT";
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
			ss << "ALGO_3";
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
			ss << "WINOGRAD_NONFUSED";
			break;
	}

	return ss.str();
}
