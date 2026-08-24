#include "alveare/npy.h"
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>

namespace alveare {

NpyArray load_npy(const std::string& path) {
    NpyArray arr{};
    arr.fd = open(path.c_str(), O_RDONLY);
    if (arr.fd < 0) {
        throw std::runtime_error("Failed to open " + path);
    }

    struct stat st;
    if (fstat(arr.fd, &st) < 0) {
        close(arr.fd);
        throw std::runtime_error("Failed to stat " + path);
    }
    arr.mapped_size = st.st_size;

    arr.mapped_ptr = mmap(nullptr, arr.mapped_size, PROT_READ, MAP_PRIVATE, arr.fd, 0);
    if (arr.mapped_ptr == MAP_FAILED) {
        close(arr.fd);
        throw std::runtime_error("Failed to mmap " + path);
    }

    const char* ptr = static_cast<const char*>(arr.mapped_ptr);
    if (std::memcmp(ptr, "\x93NUMPY", 6) != 0) {
        munmap(arr.mapped_ptr, arr.mapped_size);
        close(arr.fd);
        throw std::runtime_error("Not a valid NPY file");
    }

    uint16_t header_len;
    std::memcpy(&header_len, ptr + 8, 2);

    arr.data = static_cast<void*>(static_cast<char*>(arr.mapped_ptr) + 10 + header_len);
    arr.data_size = arr.mapped_size - (10 + header_len);

    std::string header(ptr + 10, header_len);
    size_t dtype_pos = header.find("'descr':");
    if (dtype_pos != std::string::npos) {
        size_t quote_start = header.find('\'', dtype_pos + 8);
        if (quote_start != std::string::npos) {
            size_t quote_end = header.find('\'', quote_start + 1);
            if (quote_end != std::string::npos) {
                arr.dtype = header.substr(quote_start + 1, quote_end - quote_start - 1);
            }
        }
    }

    return arr;
}

void free_npy(NpyArray& arr) {
    if (arr.mapped_ptr) {
        munmap(arr.mapped_ptr, arr.mapped_size);
        arr.mapped_ptr = nullptr;
    }
    if (arr.fd >= 0) {
        close(arr.fd);
        arr.fd = -1;
    }
}

#if defined(__F16C__) || defined(__AVX2__)
#include <immintrin.h>
#endif

static inline float fp16_to_fp32_fallback(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp == 0) {
        if (mant == 0) {
            float f = 0.0f;
            uint32_t u = sign;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x03FF;
    } else if (exp == 31) {
        uint32_t u = sign | 0x7F800000 | (mant << 13);
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }

    exp = exp + (127 - 15);
    uint32_t u = sign | (exp << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<float> load_float_npy(const std::string& path) {
    NpyArray arr;
    try {
        arr = load_npy(path);
    } catch (const std::exception& e) {
        return {};
    }
    if (!arr.data) {
        return {};
    }
    
    std::vector<float> vec;
    bool is_f16 = (arr.dtype == "<f2" || arr.dtype == "float16" || arr.dtype == "|f2");
    bool is_bf16 = (arr.dtype == "<bfloat16" || arr.dtype == "bfloat16");
    if (is_f16) {
        size_t expected_elements = arr.data_size / 2;
        vec.resize(expected_elements);
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(arr.data);
        for (size_t i = 0; i < expected_elements; ++i) {
            vec[i] = fp16_to_fp32_fallback(ptr[i]);
        }
    } else if (is_bf16) {
        size_t expected_elements = arr.data_size / 2;
        vec.resize(expected_elements);
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(arr.data);
        for (size_t i = 0; i < expected_elements; ++i) {
            uint32_t u32 = static_cast<uint32_t>(ptr[i]) << 16;
            float f;
            std::memcpy(&f, &u32, sizeof(float));
            vec[i] = f;
        }
    } else {
        vec.resize(arr.data_size / sizeof(float));
        std::memcpy(vec.data(), arr.data, arr.data_size);
    }
    
    free_npy(arr);
    return vec;
}

std::vector<uint16_t> load_uint16_npy(const std::string& path) {
    NpyArray arr;
    try {
        arr = load_npy(path);
    } catch (const std::exception& e) {
        return {};
    }
    if (!arr.data) {
        return {};
    }
    std::vector<uint16_t> vec(arr.data_size / sizeof(uint16_t));
    std::memcpy(vec.data(), arr.data, arr.data_size);
    free_npy(arr);
    return vec;
}

} // namespace alveare
