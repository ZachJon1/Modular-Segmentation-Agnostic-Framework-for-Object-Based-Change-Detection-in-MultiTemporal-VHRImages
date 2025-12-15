#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

// Minimal PNG loader (stb_image)
#include "lodepng.h"

// Simple CUDA error check helper
#define CUDA_CHECK(stmt)                                                                          \
    do {                                                                                          \
        cudaError_t err = (stmt);                                                                 \
        if (err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) +      \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                                         \
    } while (0)

struct DirectionalRCM {
    double overlap{0.0};
    double fragmentation{0.0};
    double composite{0.0};
};

struct SymmetricRCM {
    double overlap{0.0};
    double fragmentation{0.0};
    double composite{0.0};
};

struct RCMResults {
    DirectionalRCM forward;
    DirectionalRCM backward;
    SymmetricRCM symmetric;
};

__global__ void build_intersection(const int32_t* a, const int32_t* b, std::size_t n,
                                   std::uint64_t* matrix, int n_b) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t ai = a[idx];
        int32_t bi = b[idx];
        std::size_t flat = static_cast<std::size_t>(ai) * n_b + static_cast<std::size_t>(bi);
        atomicAdd(reinterpret_cast<unsigned long long*>(&matrix[flat]), 1ULL);
    }
}

DirectionalRCM compute_directional(const std::vector<std::uint64_t>& mat, int n_a, int n_b) {
    std::vector<double> row_sums(n_a, 0.0);
    double total = 0.0;
    for (int i = 0; i < n_a; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n_b; ++j) {
            row_sum += static_cast<double>(mat[i * n_b + j]);
        }
        row_sums[i] = row_sum;
        total += row_sum;
    }
    if (total == 0.0) {
        throw std::runtime_error("Empty intersection matrix encountered.");
    }
    double overlap = 0.0;
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < n_a; ++i) {
        double max_row = 0.0;
        int parts = 0;
        for (int j = 0; j < n_b; ++j) {
            double val = static_cast<double>(mat[i * n_b + j]) / total;
            if (val > 0.0) {
                ++parts;
            }
            if (val > max_row) {
                max_row = val;
            }
        }
        double w_i = row_sums[i] / total;
        double m_i = (row_sums[i] > 0.0) ? max_row / w_i : 0.0;
        numerator += w_i * (parts - 1) * (1.0 - m_i);
        denominator += w_i * (parts - 1);
        overlap += max_row;
    }
    double fragmentation = (denominator > 0.0) ? numerator / denominator : 0.0;
    overlap = std::min(std::max(overlap, 0.0), 1.0);
    fragmentation = std::min(std::max(fragmentation, 0.0), 1.0);
    double composite = 0.5 * fragmentation + 0.5 * (1.0 - overlap);
    return DirectionalRCM{overlap, fragmentation, composite};
}

RCMResults compute_rcm_cpu(const std::vector<std::uint64_t>& mat, int n_a, int n_b) {
    DirectionalRCM forward = compute_directional(mat, n_a, n_b);

    // transpose view without copying
    std::vector<std::uint64_t> mat_T(static_cast<std::size_t>(n_a) * n_b);
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_b; ++j) {
            mat_T[j * n_a + i] = mat[i * n_b + j];
        }
    }
    DirectionalRCM backward = compute_directional(mat_T, n_b, n_a);

    SymmetricRCM sym{};
    sym.overlap = 0.5 * (forward.overlap + backward.overlap);
    sym.fragmentation = 0.5 * (forward.fragmentation + backward.fragmentation);
    sym.composite = 0.5 * (forward.composite + backward.composite);
    return RCMResults{forward, backward, sym};
}

struct Mask {
    int width{0};
    int height{0};
    std::vector<uint32_t> data;
};

Mask load_mask(const std::string& path) {
    std::vector<unsigned char> image;
    unsigned width = 0, height = 0;
    unsigned err = lodepng::decode(image, width, height, path, LCT_GREY, 16);
    if (err) {
        throw std::runtime_error("Failed to load image: " + path + " error: " +
                                 std::string(lodepng_error_text(err)));
    }
    std::vector<uint32_t> data(static_cast<std::size_t>(width) * height);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (image.size() == data.size() * 2) {
            data[i] = static_cast<uint32_t>(image[2 * i] | (image[2 * i + 1] << 8));
        } else {
            data[i] = static_cast<uint32_t>(image[i]);
        }
    }
    return Mask{static_cast<int>(width), static_cast<int>(height), std::move(data)};
}

int relabel_inplace(std::vector<int32_t>& vals) {
    std::unordered_map<int32_t, int32_t> remap;
    remap.reserve(vals.size() / 4 + 1);
    int32_t next = 0;
    for (int32_t v : vals) {
        if (remap.find(v) == remap.end()) {
            remap.emplace(v, next++);
        }
    }
    for (auto& v : vals) {
        v = remap[v];
    }
    return next;
}

std::vector<std::uint64_t> compute_intersection_gpu(const std::vector<int32_t>& h_a,
                                                    const std::vector<int32_t>& h_b, int n_a,
                                                    int n_b, double& ms_kernel) {
    if (h_a.size() != h_b.size()) {
        throw std::runtime_error("Segmentation masks must share spatial dimensions.");
    }
    std::size_t n = h_a.size();

    int32_t* d_a = nullptr;
    int32_t* d_b = nullptr;
    std::uint64_t* d_matrix = nullptr;
    std::size_t matrix_elems = static_cast<std::size_t>(n_a) * n_b;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_elems * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(d_matrix, 0, matrix_elems * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    auto start = std::chrono::high_resolution_clock::now();
    build_intersection<<<blocks, threads>>>(d_a, d_b, n, d_matrix, n_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    ms_kernel = std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<std::uint64_t> h_matrix(matrix_elems);
    CUDA_CHECK(cudaMemcpy(h_matrix.data(), d_matrix, matrix_elems * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_matrix));
    return h_matrix;
}

void save_matrix_csv(const std::string& path, const std::vector<std::uint64_t>& mat, int n_a,
                     int n_b) {
    std::ofstream out(path);
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_b; ++j) {
            out << mat[i * n_b + j];
            if (j + 1 < n_b) out << ",";
        }
        out << "\n";
    }
}

void save_metrics_csv(const std::string& path, const RCMResults& res, double ms_gpu,
                      double ms_cpu, double ms_kernel, double gflops_est) {
    std::ofstream out(path);
    out << "direction,overlap,fragmentation,composite\n";
    out << "forward," << res.forward.overlap << "," << res.forward.fragmentation << ","
        << res.forward.composite << "\n";
    out << "backward," << res.backward.overlap << "," << res.backward.fragmentation << ","
        << res.backward.composite << "\n";
    out << "symmetric," << res.symmetric.overlap << "," << res.symmetric.fragmentation << ","
        << res.symmetric.composite << "\n";
    out << "\n";
    out << "timing_ms_gpu," << ms_gpu << "\n";
    out << "timing_ms_kernel," << ms_kernel << "\n";
    out << "timing_ms_cpu," << ms_cpu << "\n";
    out << "gflops_est," << gflops_est << "\n";
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: rcm_metrics_cuda <mask_a.png> <mask_b.png> <metrics.csv> "
                     "<matrix.csv>\n";
        return 1;
    }
    std::string mask_a_path = argv[1];
    std::string mask_b_path = argv[2];
    std::string metrics_out = argv[3];
    std::string matrix_out = argv[4];

    try {
        Mask mask_a = load_mask(mask_a_path);
        Mask mask_b = load_mask(mask_b_path);
        if (mask_a.width != mask_b.width || mask_a.height != mask_b.height) {
            throw std::runtime_error("Segmentation masks must share spatial dimensions.");
        }

        std::size_t n = static_cast<std::size_t>(mask_a.width) * mask_a.height;
        std::vector<int32_t> h_a(n), h_b(n);
        for (std::size_t i = 0; i < n; ++i) {
            h_a[i] = static_cast<int32_t>(mask_a.data[i]);
            h_b[i] = static_cast<int32_t>(mask_b.data[i]);
        }

        int n_a = relabel_inplace(h_a);
        int n_b = relabel_inplace(h_b);
        double ms_kernel = 0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::uint64_t> mat_gpu =
            compute_intersection_gpu(h_a, h_b, n_a, n_b, ms_kernel);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms_gpu_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto cpu_start = std::chrono::high_resolution_clock::now();
        std::vector<std::uint64_t> mat_cpu(static_cast<std::size_t>(n_a) * n_b, 0);
        for (int r = 0; r < mask_a.height; ++r) {
            for (int c = 0; c < mask_a.width; ++c) {
                std::size_t idx =
                    static_cast<std::size_t>(h_a[static_cast<std::size_t>(r) * mask_a.width + c]) *
                        n_b +
                    h_b[static_cast<std::size_t>(r) * mask_b.width + c];
                mat_cpu[idx] += 1;
            }
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto ms_cpu = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        RCMResults res = compute_rcm_cpu(mat_gpu, n_a, n_b);

        // Estimate GFLOPs as ~3 operations per pixel processed on GPU
        double pixels = static_cast<double>(mask_a.width) * mask_a.height;
        double gflops = (ms_kernel > 0.0) ? (3.0 * pixels) / (ms_kernel * 1e6) : 0.0;

        save_metrics_csv(metrics_out, res, ms_gpu_total, ms_cpu, ms_kernel, gflops);
        save_matrix_csv(matrix_out, mat_gpu, n_a, n_b);

        std::cout << "Forward overlap: " << res.forward.overlap
                  << " fragmentation: " << res.forward.fragmentation
                  << " composite: " << res.forward.composite << "\n";
        std::cout << "Backward overlap: " << res.backward.overlap
                  << " fragmentation: " << res.backward.fragmentation
                  << " composite: " << res.backward.composite << "\n";
        std::cout << "Symmetric overlap: " << res.symmetric.overlap
                  << " fragmentation: " << res.symmetric.fragmentation
                  << " composite: " << res.symmetric.composite << "\n";
        std::cout << "GPU total ms: " << ms_gpu_total << " (kernel " << ms_kernel
                  << "), CPU intersection ms: " << ms_cpu << ", est. GFLOPs: " << gflops << "\n";
        std::cout << "Matrix saved to: " << matrix_out << "\n";
        std::cout << "Metrics saved to: " << metrics_out << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
