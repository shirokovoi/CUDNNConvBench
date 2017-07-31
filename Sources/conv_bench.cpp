#include <cudnn_helper.hpp>
#include <Types.hpp>
#include <string>
#include <vector>
#include <cudnn.h>
#include <chrono>
#include <algorithm>
#include <fstream>

const float alpha = 1.0f;
const float beta = 0.0f;
const int repeats = 300;

bool ForwardPass(cudnnHandle_t handle,
                 ConvolutionDescriptor<float> convolution_descriptor,
                 TensorDescriptor4d<float> input_descriptor,
                 Tensor<float> input,
                 FilterDescriptor4d<float> filter_descriptor,
                 Tensor<float> filter,
                 TensorDescriptor4d<float> output_descriptor,
                 Tensor<float> output,
                 cudnnConvolutionFwdAlgo_t algo,
                 size_t workspace_size,
                 Tensor<float> workspace)
{
	cudnnStatus_t status = cudnnConvolutionForward(handle, &alpha,
	                                               input_descriptor.desc(), input.begin(),
	                                               filter_descriptor.desc(), filter.begin(),
	                                               convolution_descriptor.desc(),
	                                               algo, workspace.begin(), workspace_size,
	                                               &beta,
	                                               output_descriptor.desc(), output.begin());

	return status == CUDNN_STATUS_SUCCESS;
}

bool BackwardFilterPass(cudnnHandle_t handle,
                        ConvolutionDescriptor<float> convolution_descriptor,
                        TensorDescriptor4d<float> input_descriptor,
                        Tensor<float> input,
                        TensorDescriptor4d<float> delta_descriptor,
                        Tensor<float> delta,
                        FilterDescriptor4d<float> dW_descriptor,
                        Tensor<float> dW,
                        cudnnConvolutionBwdFilterAlgo_t algo,
                        size_t workspace_size,
                        Tensor<float> workspace)
{
	cudnnStatus_t status = cudnnConvolutionBackwardFilter(handle, &alpha,
	                                                      input_descriptor.desc(), input.begin(),
	                                                      delta_descriptor.desc(), delta.begin(),
	                                                      convolution_descriptor.desc(),
	                                                      algo, workspace.begin(), workspace_size,
	                                                      &beta,
	                                                      dW_descriptor.desc(), dW.begin());
	return status == CUDNN_STATUS_SUCCESS;
}

bool BackwardDataPass(cudnnHandle_t handle,
                      ConvolutionDescriptor<float> convolution_descriptor,
                      FilterDescriptor4d<float> filter_descriptor,
                      Tensor<float> filter,
                      TensorDescriptor4d<float> delta_descriptor,
                      Tensor<float> delta,
                      TensorDescriptor4d<float> dX_descriptor,
                      Tensor<float> dX,
                      cudnnConvolutionBwdDataAlgo_t algo,
                      size_t workspace_size,
                      Tensor<float> workspace)
{
	cudnnStatus_t status = cudnnConvolutionBackwardData(handle, &alpha,
	                                                    filter_descriptor.desc(), filter.begin(),
	                                                    delta_descriptor.desc(), delta.begin(),
	                                                    convolution_descriptor.desc(),
	                                                    algo, workspace.begin(), workspace_size,
	                                                    &beta,
	                                                    dX_descriptor.desc(), dX.begin());

	return status == CUDNN_STATUS_SUCCESS;
}

Result Bench(Problem problem, curandGenerator_t curand_gen)
{
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
	TensorDescriptor4d<float> input_descriptor(format, problem.m_BatchSize, problem.m_InputMaps, problem.m_InputSize.m_Height, problem.m_InputSize.m_Width);
	FilterDescriptor4d<float> filter_descriptor(format, problem.m_OutputMaps, problem.m_InputMaps, problem.m_FilterSize.m_Height, problem.m_FilterSize.m_Width);
	ConvolutionDescriptor<float> convolulion_descriptor(problem.m_Padding.m_Height, problem.m_Padding.m_Width, problem.m_Stride.m_Height, problem.m_Stride.m_Width);

	Result result;
	result.m_Problem = problem;
	result.m_Repeats = repeats;

	Size output_size;
	int output_images;
	int output_maps;

	cudnnGetConvolution2dForwardOutputDim(convolulion_descriptor.desc(),
	                                      input_descriptor.desc(),
	                                      filter_descriptor.desc(),
	                                      &output_images,
	                                      &output_maps,
	                                      &(output_size.m_Height),
	                                      &(output_size.m_Width));

	TensorDescriptor4d<float> output_descriptor(format, output_images, output_maps, output_size.m_Height, output_size.m_Width);

	CudnnHandle handle;

	cudnnConvolutionFwdAlgoPerf_t fwd_algos[32];
	int fwd_algos_number = 0;
	cudnnFindConvolutionForwardAlgorithm(handle.handle(),
	                                     input_descriptor.desc(),
	                                     filter_descriptor.desc(),
	                                     convolulion_descriptor.desc(),
	                                     output_descriptor.desc(),
	                                     32,
	                                     &fwd_algos_number,
	                                     fwd_algos);
	for (int fwd_index = 0; fwd_index < fwd_algos_number; fwd_index++)
	{
		cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algos[fwd_index].algo;
		size_t fwd_workspace_size;
		cudnnStatus_t rc = cudnnGetConvolutionForwardWorkspaceSize(handle.handle(),
		                                                           input_descriptor.desc(),
		                                                           filter_descriptor.desc(),
		                                                           convolulion_descriptor.desc(),
		                                                           output_descriptor.desc(),
		                                                           fwd_algo,
		                                                           &fwd_workspace_size);
		if (rc != CUDNN_STATUS_SUCCESS)
		{
			std::cerr << problem.ToString() << " with fwd = " << FwdAlgo_ToString(fwd_algo) << " is skipped (rc = " << rc << ")" << std::endl;
			continue;
		}

		Tensor<float> fwd_workspace = zeros<float>(std::vector<int>{(int)(fwd_workspace_size / sizeof(float)), 1});
		Tensor<float> input = rand<float>(std::vector<int>{problem.m_InputSize.m_Width, problem.m_InputSize.m_Height, problem.m_BatchSize, problem.m_InputMaps}, curand_gen);
		Tensor<float> output = rand<float>(std::vector<int>{output_size.m_Width , output_size.m_Height, output_images, output_maps}, curand_gen);
		Tensor<float> filter = rand<float>(std::vector<int>{problem.m_FilterSize.m_Width, problem.m_FilterSize.m_Height, problem.m_InputMaps, problem.m_OutputMaps}, curand_gen);

		if (!ForwardPass(handle.handle(), convolulion_descriptor,
		                 input_descriptor, input,
		                 filter_descriptor, filter,
		                 output_descriptor, output,
		                 fwd_algo, fwd_workspace_size, fwd_workspace))
		{
			std::cerr << problem.ToString() << " with fwd_algo = " << fwd_algo << " is skipped (fwd pass is failed)" << std::endl;
			continue;
		}

		cudaDeviceSynchronize();

		auto start = std::chrono::steady_clock::now();
		for (int iteration = 0; iteration < repeats; iteration++)
			ForwardPass(handle.handle(), convolulion_descriptor,
			            input_descriptor, input,
			            filter_descriptor, filter,
			            output_descriptor, output,
			            fwd_algo, fwd_workspace_size, fwd_workspace);

		cudaDeviceSynchronize();
		auto fwd_elapsed = std::chrono::steady_clock::now() - start;
		auto fwd_avg = (int)(std::chrono::duration_cast<std::chrono::microseconds>(fwd_elapsed).count() / repeats);

		result.m_ForwardTimes.emplace(fwd_algo, fwd_avg);
	}

	cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algos[32];
	int bwd_data_algos_number = 0;
	cudnnFindConvolutionBackwardDataAlgorithm(handle.handle(),
	                                          filter_descriptor.desc(),
	                                          output_descriptor.desc(),
	                                          convolulion_descriptor.desc(),
	                                          input_descriptor.desc(),
	                                          32,
	                                          &bwd_data_algos_number,
	                                          bwd_data_algos);
	for (int bwd_data_index = 0; bwd_data_index < bwd_data_algos_number; bwd_data_index++)
	{
		cudnnConvolutionBwdDataAlgo_t bwd_data_algo = bwd_data_algos[bwd_data_index].algo;
		size_t bwd_data_workspace_size;
		cudnnStatus_t rc = cudnnGetConvolutionBackwardDataWorkspaceSize(handle.handle(),
		                                                  filter_descriptor.desc(),
		                                                  output_descriptor.desc(),
		                                                  convolulion_descriptor.desc(),
		                                                  input_descriptor.desc(),
		                                                  bwd_data_algo,
		                                                  &bwd_data_workspace_size);
		if (rc != CUDNN_STATUS_SUCCESS)
		{
			std::cerr << problem.ToString() << " with bwd_data = " << BwdData_ToString(bwd_data_algo) << " is skipped (rc = " << rc << ")" << std::endl;
			continue;
		}

		Tensor<float> bwd_data_workspace = zeros<float>(std::vector<int>{(int)(bwd_data_workspace_size / sizeof(float)), 1});
		Tensor<float> input = rand<float>(std::vector<int>{problem.m_InputSize.m_Width, problem.m_InputSize.m_Height, problem.m_BatchSize, problem.m_InputMaps}, curand_gen);
		Tensor<float> output = rand<float>(std::vector<int>{output_size.m_Width , output_size.m_Height, output_images, output_maps}, curand_gen);
		Tensor<float> filter = rand<float>(std::vector<int>{problem.m_FilterSize.m_Width, problem.m_FilterSize.m_Height, problem.m_InputMaps, problem.m_OutputMaps}, curand_gen);
						if (!BackwardDataPass(handle.handle(), convolulion_descriptor,
				                 filter_descriptor, filter,
				                 output_descriptor, output,
				                 input_descriptor, input,
				                 bwd_data_algo, bwd_data_workspace_size, bwd_data_workspace))
				{
					std::cerr << problem.ToString() << " with bwd_data_algo = " << bwd_data_algo << " is skipped (bwd data pass is failed)" << std::endl;
					continue;
				}
		cudaDeviceSynchronize();

		auto start = std::chrono::steady_clock::now();
		for (int iteration = 0; iteration < repeats; iteration++)
			BackwardDataPass(handle.handle(), convolulion_descriptor,
			                 filter_descriptor, filter,
			                 output_descriptor, output,
			                 input_descriptor, input,
			                 bwd_data_algo, bwd_data_workspace_size, bwd_data_workspace);
		cudaDeviceSynchronize();

		auto bwd_data_elapsed = std::chrono::steady_clock::now() - start;
		auto bwd_data_avg = (int)(std::chrono::duration_cast<std::chrono::microseconds>(bwd_data_elapsed).count()/repeats);

		result.m_BackwardDataTimes.emplace(bwd_data_algo, bwd_data_avg);
	}

	cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algos[32];
	int bwd_filter_algos_number = 0;
	cudnnFindConvolutionBackwardFilterAlgorithm(handle.handle(),
	                                            input_descriptor.desc(),
	                                            output_descriptor.desc(),
	                                            convolulion_descriptor.desc(),
	                                            filter_descriptor.desc(),
	                                            32,
	                                            &bwd_filter_algos_number,
	                                            bwd_filter_algos);
	for (int bwd_filter_index = 0; bwd_filter_index < bwd_filter_algos_number; bwd_filter_index++)
	{
		cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo = bwd_filter_algos[bwd_filter_index].algo;
		size_t bwd_filter_workspace_size;

		cudnnStatus_t rc = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.handle(),
		                                                                  input_descriptor.desc(),
		                                                                  output_descriptor.desc(),
		                                                                  convolulion_descriptor.desc(),
		                                                                  filter_descriptor.desc(),
		                                                                  bwd_filter_algo,
		                                                                  &bwd_filter_workspace_size);
		if (rc != CUDNN_STATUS_SUCCESS)
		{
			std::cerr << problem.ToString() << " with bwd_filter = " << BwdFilter_ToString(bwd_filter_algo) << " is skipped (rc = " << rc << ")" << std::endl;
			continue;
		}

		Tensor<float> bwd_filter_workspace = zeros<float>(std::vector<int>{(int)(bwd_filter_workspace_size / sizeof(float)), 1});
		Tensor<float> input = rand<float>(std::vector<int>{problem.m_InputSize.m_Width, problem.m_InputSize.m_Height, problem.m_BatchSize, problem.m_InputMaps}, curand_gen);
		Tensor<float> output = rand<float>(std::vector<int>{output_size.m_Width , output_size.m_Height, output_images, output_maps}, curand_gen);
		Tensor<float> filter = rand<float>(std::vector<int>{problem.m_FilterSize.m_Width, problem.m_FilterSize.m_Height, problem.m_InputMaps, problem.m_OutputMaps}, curand_gen);

						if (!BackwardFilterPass(handle.handle(), convolulion_descriptor,
				                        input_descriptor, input,
				                        output_descriptor, output,
				                        filter_descriptor, filter,
				                        bwd_filter_algo, bwd_filter_workspace_size, bwd_filter_workspace))
				{
					std::cerr << problem.ToString() << " with bwd_filter_algo = " << bwd_filter_algo << " is skipped (bwd filter pass is failed)" << std::endl;
					continue;
				}

		cudaDeviceSynchronize();

		auto start = std::chrono::steady_clock::now();
		for (int iteration = 0; iteration < repeats; iteration++)
			BackwardFilterPass(handle.handle(), convolulion_descriptor,
			                   input_descriptor, input,
			                   output_descriptor, output,
			                   filter_descriptor, filter,
			                   bwd_filter_algo, bwd_filter_workspace_size, bwd_filter_workspace);

		cudaDeviceSynchronize();
		auto bwd_filter_elapsed = std::chrono::steady_clock::now() - start;
		auto bwd_filter_avg = (int)(std::chrono::duration_cast<std::chrono::microseconds>(bwd_filter_elapsed).count() / repeats);
		result.m_BackwardFilterTimes.emplace(bwd_filter_algo, bwd_filter_avg);
	}

	return result;
}

int main(int argc, char** argv)
{
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, 123ULL);

	std::string input_path = "./in.csv";
	if (argc > 1)
		input_path = argv[1];

	std::string output_path = "./out.csv";
	if (argc > 2)
		output_path = argv[2];

	std::ifstream input_file(input_path);
	if (!input_file.is_open())
	{
		std::cerr << "Can't open input file (" << input_path << ")" << std::endl;
		return -1;
	}

	std::ofstream output_file(output_path);
	if (!output_file.is_open())
	{
		std::cerr << "Can't open output file (" << output_path << ")" << std::endl;
		return -1;
	}

	std::vector<Problem> problems;
	std::string line;
	while (std::getline(input_file, line))
	{
		Problem p;
		if (!p.Parse(line))
		{
			std::cerr << "Error parse line \"" << line << "\"" << std::endl;
			return -1;
		}
		problems.emplace_back(p);
	}
	std::cerr << "Read " << problems.size() << " problems" << std::endl;

	std::vector<Result> results;
	auto start = std::chrono::steady_clock::now();
	for (auto& problem: problems)
	{
		results.emplace_back(Bench(problem, generator));
	}
	auto stop = std::chrono::steady_clock::now();
	std::cerr << "Test takes " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " s" << std::endl;

	for (auto& result: results)
	{
		output_file << result.ToString() << std::endl;
	}

	return 0;
}