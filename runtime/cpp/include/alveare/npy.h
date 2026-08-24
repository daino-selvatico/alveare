#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace alveare {
struct NpyArray {
    std::vector<size_t> shape;
    std::string dtype;
    bool fortran_order;
    void* data;
    size_t data_size;
    void* mapped_ptr;
    size_t mapped_size;
    int fd;
};

NpyArray load_npy(const std::string& path);
void free_npy(NpyArray& arr);
std::vector<float> load_float_npy(const std::string& path);
std::vector<uint16_t> load_uint16_npy(const std::string& path);
} // namespace alveare
