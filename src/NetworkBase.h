#ifndef NETWORK_BASE_H_INCLUDED
#define NETWORK_BASE_H_INCLUDED

#include "config.h"

#include <vector>
#include <cmath>
#include <cassert>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif

#include "Im2Col.h"

namespace NetworkBase
{


// Winograd filter transformation changes 3x3 filters to 4x4
constexpr auto WINOGRAD_ALPHA = 4;
constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

void softmax(const std::vector<float> &input,
             std::vector<float> &output,
             float temperature = 1.0f);

void process_bn_var(std::vector<float> &weights,
                    const float epsilon = 1e-5f);

std::vector<float> winograd_transform_f(const std::vector<float> &f,
                                        const int outputs, const int channels);
std::vector<float> zeropad_U(const std::vector<float> &U,
                             const int outputs, const int channels,
                             const int outputs_pad, const int channels_pad);
void winograd_transform_in(const std::vector<float> &in,
                           std::vector<float> &V,
                           const int C);
void winograd_transform_out(const std::vector<float> &M,
                            std::vector<float> &Y,
                            const int K);
void winograd_convolve3(const int outputs,
                        const std::vector<float> &input,
                        const std::vector<float> &U,
                        std::vector<float> &V,
                        std::vector<float> &M,
                        std::vector<float> &output);
void winograd_sgemm(const std::vector<float> &U,
                    std::vector<float> &V,
                    std::vector<float> &M, const int C, const int K);
void compare_net_outputs(std::vector<float> &data,
                         std::vector<float> &ref);
};


template <unsigned int filter_size,
          unsigned int height,
          unsigned int width>
void convolve(size_t outputs,
              const std::vector<net_t> &input,
              const std::vector<float> &weights,
              const std::vector<float> &biases,
              std::vector<float> &output)
{
    // The size of the board is defined at compile time
    constexpr unsigned int board_squares = width * height;
    constexpr unsigned int filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * board_squares == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size, height, width>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, board_squares, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], board_squares,
                0.0f, &output[0], board_squares);

    for (unsigned int o = 0; o < outputs; o++)
    {
        for (unsigned int b = 0; b < board_squares; b++)
        {
            output[(o * board_squares) + b] =
                biases[o] + output[(o * board_squares) + b];
        }
    }
}

template <unsigned int inputs,
          unsigned int outputs,
          size_t W, size_t B>
inline void innerproduct(const std::vector<float> &input,
                         const std::array<float, W> &weights,
                         const std::array<float, B> &biases,
                         std::vector<float> &output)
{
    assert(B == outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    auto lambda_ReLU = [](float val) { return (val > 0.0f) ? val : 0.0f; };

    for (unsigned int o = 0; o < outputs; o++)
    {
        float val = biases[o] + output[o];
        if (outputs == 256)
        {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }
}

template <size_t spatial_size>
inline void batchnorm(size_t channels,
                      std::vector<float> &data,
                      const float *means,
                      const float *stddivs,
                      const float *eltwise = nullptr)
{
    auto lambda_ReLU = [](float val) { return (val > 0.0f) ? val : 0.0f; };

    for (auto c = size_t{0}; c < channels; ++c)
    {
        auto mean = means[c];
        auto scale_stddiv = stddivs[c];

        if (eltwise == nullptr)
        {
            // Classical BN
            auto arr = &data[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++)
            {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        }
        else
        {
            // BN + residual add
            auto arr = &data[c * spatial_size];
            auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++)
            {
                arr[b] = lambda_ReLU(res[b] +
                                     (scale_stddiv * (arr[b] - mean)));
            }
        }
    }
}

template <typename T>
T relative_difference(T a, T b)
{
    // Handle NaN
    if (std::isnan(a) || std::isnan(b))
    {
        return std::numeric_limits<T>::max();
    }

    constexpr auto small_number = 1e-3f;
    auto fa = std::fabs(a);
    auto fb = std::fabs(b);

    if (fa > small_number && fb > small_number)
    {
        // Handle sign difference
        if (((a < 0) != (b < 0)) && (a != 0) && (b != 0))
        {
            return std::numeric_limits<T>::max();
        }
    }

    // Handle underflow
    fa = std::max(fa, small_number);
    fb = std::max(fb, small_number);

    return std::max(fabs((fa - fb) / fa), fabs((fa - fb) / fb));
}

#endif