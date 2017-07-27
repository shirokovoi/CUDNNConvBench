//
// Created by oleg on 23.07.17.
//

#ifndef CUDNNCONVBENCH_CUDNN_HELPER_HPP
#define CUDNNCONVBENCH_CUDNN_HELPER_HPP

#include <cudnn.h>
#include <memory>
#include <sstream>
#include <vector>
#include <numeric>

#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

void throw_cudnn_err(cudnnStatus_t status, int line, const char* filename) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "CUDNN failure: " << cudnnGetErrorString(status) <<
              " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}
#define CHECK_CUDNN_ERROR(status) throw_cudnn_err(status, __LINE__, __FILE__)

class CudnnHandle {
    std::shared_ptr<cudnnHandle_t> handle_;

    struct CudnnHandleDeleter {
        void operator()(cudnnHandle_t * handle) {
            cudnnDestroy(*handle);
            delete handle;
        }
    };

public:
    CudnnHandle() : handle_(new cudnnHandle_t, CudnnHandleDeleter()) {
        CHECK_CUDNN_ERROR(cudnnCreate(handle_.get()));
    }

    cudnnHandle_t handle() const { return *handle_; };
};

template<typename T>
class TensorDescriptor4d {
	std::shared_ptr<cudnnTensorDescriptor_t> desc_;

	struct TensorDescriptor4dDeleter {
		void operator()(cudnnTensorDescriptor_t * desc) {
			cudnnDestroyTensorDescriptor(*desc);
			delete desc;
		}
	};

public:

	TensorDescriptor4d() {}
	TensorDescriptor4d(const cudnnTensorFormat_t tensor_format,
	                   const int n, const int c, const int h, const int w) {
		cudnnDataType_t type;
		if (std::is_same<T, float>::value) {
			type = CUDNN_DATA_FLOAT;
#if CUDNN_MAJOR >= 6
			} else if (std::is_same<T, uint8_t>::value) {
            type = CUDNN_DATA_INT8;
#endif
		} else if (std::is_same<T, uint16_t>::value) {
			type = CUDNN_DATA_HALF;
		} else {
			throw std::runtime_error("Unknown type in TensorDescriptor4d");
		}

		cudnnTensorDescriptor_t * desc = new cudnnTensorDescriptor_t;
		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(desc));
		CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*desc,
		                                             tensor_format,
		                                             type,
		                                             n,
		                                             c,
		                                             h,
		                                             w));

		desc_.reset(desc, TensorDescriptor4dDeleter());
	}

	cudnnTensorDescriptor_t desc() const { return *desc_; }

};

template<typename T>
class FilterDescriptor4d {
	std::shared_ptr<cudnnFilterDescriptor_t> desc_;

	struct FilterDescriptor4dDeleter {
		void operator()(cudnnFilterDescriptor_t * desc) {
			cudnnDestroyFilterDescriptor(*desc);
			delete desc;
		}
	};

public:
	FilterDescriptor4d() {}

	FilterDescriptor4d(const cudnnTensorFormat_t tensor_format,
	                   int k, int c, int h, int w) {
		cudnnDataType_t type;
		if (std::is_same<T, float>::value) {
			type = CUDNN_DATA_FLOAT;
#if CUDNN_MAJOR >= 6
			} else if (std::is_same<T, uint8_t>::value) {
            type = CUDNN_DATA_INT8;
#endif
		} else if (std::is_same<T, uint16_t>::value) {
			type = CUDNN_DATA_HALF;
		} else {
			throw std::runtime_error("Unknown type in FilterDescriptor4d");
		}

		cudnnFilterDescriptor_t * desc = new cudnnFilterDescriptor_t;
		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(desc));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*desc, type, tensor_format, k, c, h, w));

		desc_.reset(desc, FilterDescriptor4dDeleter());
	}

	cudnnFilterDescriptor_t desc() const { return *desc_; }

};

template <typename T>
class ConvolutionDescriptor {
	std::shared_ptr<cudnnConvolutionDescriptor_t> desc_;

	struct ConvolutionDescriptorDeleter {
		void operator()(cudnnConvolutionDescriptor_t * desc) {
			cudnnDestroyConvolutionDescriptor(*desc);
			delete desc;
		}
	};
public:


	ConvolutionDescriptor(int pad_h, int pad_w, int hstride, int wstride) :
			desc_(new cudnnConvolutionDescriptor_t, ConvolutionDescriptorDeleter()) {

		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(desc_.get()));
#if CUDNN_MAJOR >= 6
		cudnnDataType_t type;
        if (std::is_same<T, float>::value) {
            type = CUDNN_DATA_FLOAT;
        } else if (std::is_same<T, uint8_t>::value) {
            type = CUDNN_DATA_INT8;
        } else if (std::is_same<T, uint16_t>::value) {
            type = CUDNN_DATA_HALF;
        } else if (std::is_same<T, int>::value) {
            type = CUDNN_DATA_INT32;
        } else {
            throw std::runtime_error("Unknown type in ConvolutionDescriptor");
        }


        CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*desc_,
                                                          pad_h,
                                                          pad_w,
                                                          hstride,
                                                          wstride,
                                                          1,
                                                          1,
                                                          CUDNN_CONVOLUTION,
                                                          type));
#else
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*desc_,
		                                                  pad_h,
		                                                  pad_w,
		                                                  hstride,
		                                                  wstride,
		                                                  1,
		                                                  1,
		                                                  CUDNN_CONVOLUTION));

#endif

	}

	cudnnConvolutionDescriptor_t desc() const { return *desc_; };

};

template<typename T>
class TensorDescriptorNd {
    std::shared_ptr<cudnnTensorDescriptor_t> desc_;

    struct TensorDescriptorNdDeleter {
        void operator()(cudnnTensorDescriptor_t * desc) {
            cudnnDestroyTensorDescriptor(*desc);
            delete desc;
        }
    };

public:

    TensorDescriptorNd(const std::vector<int>& dim,
                       const std::vector<int>& stride) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value)
            type = CUDNN_DATA_FLOAT;
        else if (std::is_same<T, uint16_t>::value)
            type = CUDNN_DATA_HALF;
#if CUDNN_MAJOR >= 6
        else if (std::is_same<T, uint8_t>::value)
            type = CUDNN_DATA_INT8;
#endif
        else
            throw std::runtime_error("Unknown type in TensorDescriptorNd");

        cudnnTensorDescriptor_t * desc = new cudnnTensorDescriptor_t;

        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptor(*desc, type, dim.size(),
                                                     &dim[0], &stride[0]));

        desc_.reset(desc, TensorDescriptorNdDeleter());
    }

    cudnnTensorDescriptor_t desc() const { return *desc_; }

};

template<typename T>
class TensorDescriptorNdArray {
    std::shared_ptr<cudnnTensorDescriptor_t> desc_array_;

    struct ArrayDeleter {
        int num_;
        ArrayDeleter(int num) : num_(num) {}

        void operator()(cudnnTensorDescriptor_t *desc_array) {
            for (int i = 0; i < num_; ++i) {
                cudnnDestroyTensorDescriptor(desc_array[i]);
            }

            delete[] desc_array;
        }
    };

    public:

    TensorDescriptorNdArray(std::vector<int> dim,
                            std::vector<int> stride,
                            int num) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value)
            type = CUDNN_DATA_FLOAT;
        else if (std::is_same<T, uint16_t>::value)
            type = CUDNN_DATA_HALF;
#if CUDNN_MAJOR >= 6
        else if (std::is_same<T, uint8_t>::value)
            type = CUDNN_DATA_INT8;
#endif
        else
            throw std::runtime_error("Unknown type in TensorDescriptorNdArray ");

        cudnnTensorDescriptor_t * desc_array = new cudnnTensorDescriptor_t[num];

        for (int i = 0; i < num; ++i) {
            CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_array[i]));
            CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptor(desc_array[i], type, dim.size(),
                                                         &dim[0], &stride[0]));
        }

        desc_array_.reset(desc_array, ArrayDeleter(num));
    }

    cudnnTensorDescriptor_t * ptr() const { return desc_array_.get(); }
};

template<typename T>
class FilterDescriptorNd {
    std::shared_ptr<cudnnFilterDescriptor_t> desc_;

    struct FilterDescriptorNdDeleter {
        void operator()(cudnnFilterDescriptor_t * desc) {
            cudnnDestroyFilterDescriptor(*desc);
            delete desc;
        }
    };

public:

    FilterDescriptorNd() {}

    FilterDescriptorNd(const cudnnTensorFormat_t tensor_format,
                       const std::vector<int> dim) {
        cudnnDataType_t type;
        if (std::is_same<T, float>::value)
            type = CUDNN_DATA_FLOAT;
        else if (std::is_same<T, uint16_t>::value)
            type = CUDNN_DATA_HALF;
#if CUDNN_MAJOR >= 6
        else if (std::is_same<T, uint8_t>::value)
            type = CUDNN_DATA_INT8;
#endif
        else
            throw std::runtime_error("Unknown type in FilterDescriptorNd");

        cudnnFilterDescriptor_t * desc = new cudnnFilterDescriptor_t;
        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetFilterNdDescriptor(*desc, type, tensor_format, dim.size(), &dim[0]));

        desc_.reset(desc, FilterDescriptorNdDeleter());
    }

    cudnnFilterDescriptor_t desc() { return *desc_; }
};

template <typename T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteCudaPtr {
        void operator()(T *p) const {
            cudaFree(p);
        }
    };

    std::shared_ptr<T> ptr_;

public:

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&tmp_ptr, sizeof(T) * size_);

        ptr_.reset(tmp_ptr, deleteCudaPtr());
    }

    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

template <typename T>
Tensor<T> fill(std::vector<int> dims, float val) {
     Tensor<T> tensor(dims);
     thrust::fill(thrust::device_ptr<T>(tensor.begin()),
                  thrust::device_ptr<T>(tensor.end()), val);
     return tensor;
}

template <typename T>
Tensor<T> zeros(std::vector<int> dims) {
    Tensor<T> tensor(dims);
    thrust::fill(thrust::device_ptr<T>(tensor.begin()),
                 thrust::device_ptr<T>(tensor.end()), 0.f);
    return tensor;
}

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, curandGenerator_t curand_gen) {
    Tensor<T> tensor(dims);
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    return tensor;
}

template <typename T>
typename std::enable_if<!(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, curandGenerator_t curand_gen) {

    Tensor<T> tensor(dims);
    Tensor<float> tensor_f(dims);
    curandGenerateUniform(curand_gen, tensor_f.begin(), tensor_f.size());

    thrust::copy(thrust::device_ptr<float>(tensor_f.begin()),
                 thrust::device_ptr<float>(tensor_f.end()),
                 thrust::device_ptr<T>(tensor.begin()));

    return tensor;
}

void pad_dim(int & dim) {
    if (dim % 4) {
        int pad = 4 - dim%4;
        dim += pad;
    }
}

#endif //CUDNNCONVBENCH_CUDNN_HELPER_HPP
