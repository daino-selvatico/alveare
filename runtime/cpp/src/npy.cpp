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
} // namespace alveare
