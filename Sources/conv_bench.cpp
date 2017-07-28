#include <cudnn_helper.hpp>
#include <Types.hpp>
#include <string>
#include <vector>
#include <cudnn.h>
#include <chrono>
#include <algorithm>

const float alpha = 1.0f;
const float beta = 0.0f;

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

void Bench(Problem problem, curandGenerator_t curand_gen, std::vector<Result>& out)
{
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
	TensorDescriptor4d<float> input_descriptor(format, problem.m_BatchSize, problem.m_InputMaps, problem.m_InputSize.m_Height, problem.m_InputSize.m_Width);
	FilterDescriptor4d<float> filter_descriptor(format, problem.m_OutputMaps, problem.m_InputMaps, problem.m_FilterSize.m_Height, problem.m_FilterSize.m_Width);
	ConvolutionDescriptor<float> convolulion_descriptor(problem.m_Padding.m_Height, problem.m_Padding.m_Width, problem.m_Stride.m_Height, problem.m_Stride.m_Width);

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

	for (int fwd_index = 0; fwd_index < fwd_algos_number; fwd_index++)
	{
		for (int bwd_data_index = 0; bwd_data_index < bwd_data_algos_number; bwd_data_index++)
		{
			for (int bwd_filter_index = 0; bwd_filter_index < bwd_filter_algos_number; bwd_filter_index++)
			{
				cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algos[fwd_index].algo;
				cudnnConvolutionBwdDataAlgo_t bwd_data_algo = bwd_data_algos[bwd_data_index].algo;
				cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo = bwd_filter_algos[bwd_filter_index].algo;

				size_t fwd_workspace_size;
				size_t bwd_data_workspace_size;
				size_t bwd_filter_workspace_size;

				cudnnStatus_t rc = cudnnGetConvolutionForwardWorkspaceSize(handle.handle(),
				                                                           input_descriptor.desc(),
				                                                           filter_descriptor.desc(),
				                                                           convolulion_descriptor.desc(),
				                                                           output_descriptor.desc(),
				                                                           fwd_algo,
				                                                           &fwd_workspace_size);
				if (rc != CUDNN_STATUS_SUCCESS)
				{
					std::cerr << problem.ToString() << " with fwd = " << fwd_algo << " is skipped (rc = " << rc << ")" << std::endl;
					continue;
				}

				rc = cudnnGetConvolutionBackwardDataWorkspaceSize(handle.handle(),
				                                                  filter_descriptor.desc(),
				                                                  output_descriptor.desc(),
				                                                  convolulion_descriptor.desc(),
				                                                  input_descriptor.desc(),
				                                                  bwd_data_algo,
				                                                  &bwd_data_workspace_size);
				if (rc != CUDNN_STATUS_SUCCESS)
				{
					std::cerr << problem.ToString() << " with bwd_data = " << bwd_data_algo << " is skipped (rc = " << rc << ")" << std::endl;
					continue;
				}

				rc = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.handle(),
				                                                    input_descriptor.desc(),
				                                                    output_descriptor.desc(),
				                                                    convolulion_descriptor.desc(),
				                                                    filter_descriptor.desc(),
				                                                    bwd_filter_algo,
				                                                    &bwd_filter_workspace_size);
				if (rc != CUDNN_STATUS_SUCCESS)
				{
					std::cerr << problem.ToString() << " with bwd_filter = " << bwd_filter_algo << " is skipped (rc = " << rc << ")" << std::endl;
					continue;
				}

				Tensor<float> fwd_workspace = zeros<float>(std::vector<int>{(int)(fwd_workspace_size / sizeof(float)), 1});
				Tensor<float> bwd_data_workspace = zeros<float>(std::vector<int>{(int)(bwd_data_workspace_size / sizeof(float)), 1});
				Tensor<float> bwd_filter_workspace = zeros<float>(std::vector<int>{(int)(bwd_filter_workspace_size / sizeof(float)), 1});

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
				for (int iteration = 0; iteration < 300; iteration++)
					ForwardPass(handle.handle(), convolulion_descriptor,
					            input_descriptor, input,
					            filter_descriptor, filter,
					            output_descriptor, output,
					            fwd_algo, fwd_workspace_size, fwd_workspace);

				cudaDeviceSynchronize();
				auto fwd_elapsed = std::chrono::steady_clock::now() - start;
				auto fwd_avg = (int)(std::chrono::duration_cast<std::chrono::microseconds>(fwd_elapsed).count() / 300);

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

				start = std::chrono::steady_clock::now();
				for (int iteration = 0; iteration < 300; iteration++)
					BackwardFilterPass(handle.handle(), convolulion_descriptor,
					                   input_descriptor, input,
					                   output_descriptor, output,
					                   filter_descriptor, filter,
					                   bwd_filter_algo, bwd_filter_workspace_size, bwd_filter_workspace);

				cudaDeviceSynchronize();
				auto bwd_filter_elapsed = std::chrono::steady_clock::now() - start;
				auto bwd_filter_avg = (int)(std::chrono::duration_cast<std::chrono::microseconds>(bwd_filter_elapsed).count() / 300);

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

				start = std::chrono::steady_clock::now();
				for (int iteration = 0; iteration < 300; iteration++)
					BackwardDataPass(handle.handle(), convolulion_descriptor,
					                 filter_descriptor, filter,
					                 output_descriptor, output,
					                 input_descriptor, input,
					                 bwd_data_algo, bwd_data_workspace_size, bwd_data_workspace);
				cudaDeviceSynchronize();

				auto bwd_data_elapsed = std::chrono::steady_clock::now() - start;
				auto bwd_data_avg = (int)(std::chrono::duration_cast<std::chrono::microseconds>(bwd_data_elapsed).count()/300);

				Result result;
				result.m_Repeats = 300;
				result.m_Problem = problem;
				result.m_ForwardElapsedUSec = fwd_avg;
				result.m_BackwardFilterElapsedUSec = bwd_filter_avg;
				result.m_BackwardDataElapsedUSec = bwd_data_avg;

				out.emplace_back(result);
			}
		}
	}
}

int main()
{
	return 0;
}