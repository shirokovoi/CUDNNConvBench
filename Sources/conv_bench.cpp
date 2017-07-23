#include <cudnn_helper.hpp>
#include <Types.hpp>
#include <string>
#include <vector>

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

std::vector<Result> Bench(Problem problem)
{
	std::vector<Result> result;
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
	TensorDescriptor4d<float> input_descriptor = TensorDescriptor4d<float>(format, problem.m_BatchSize, problem.m_InputMaps, problem.m_InputSize.m_Height, problem.m_InputSize.m_Width);
}

int main()
{
	return 0;
}