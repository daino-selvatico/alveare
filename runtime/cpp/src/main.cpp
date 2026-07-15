#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include "alveare/bf16.h"
#include "alveare/npy.h"
#include "alveare/config.h"

// Include XRT C++ API
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>

using namespace alveare;

std::vector<uint32_t> load_instr_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open " + path);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size))
        return buffer;
    throw std::runtime_error("Failed to read " + path);
}

int main() {
    std::cout << "Alveare C++ Runtime initializing...\n";

    try {
        unsigned int device_index = 0;
        xrt::device device(device_index);
        std::cout << "Successfully opened XRT device index " << device_index << "\n";
        
        std::string xclbin_path = "gemv_256_256.xclbin";
        std::string insts_path = "gemv_256_256.insts";
        
        std::cout << "Loading xclbin: " << xclbin_path << "\n";
        xrt::xclbin xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);

        xrt::hw_context context(device, xclbin.get_uuid());
        std::cout << "Context created successfully.\n";

        // Read NPY golden data
        std::cout << "Loading golden NPY files...\n";
        NpyArray arr_w = load_npy("W.npy");
        NpyArray arr_x = load_npy("x.npy");
        NpyArray arr_expected = load_npy("expected.npy");

        std::vector<uint32_t> instr_v = load_instr_binary(insts_path);
        
        // Kernel
        auto kernel = xrt::kernel(context, "MLIR_AIE");

        // BOs
        std::cout << "Allocating BOs...\n";
        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_w = xrt::bo(device, arr_w.data_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_x = xrt::bo(device, arr_x.data_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_y = xrt::bo(device, arr_expected.data_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        // Copy instructions
        void* buf_instr = bo_instr.map<void*>();
        std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));
        
        // Copy inputs
        void* buf_w = bo_w.map<void*>();
        std::memcpy(buf_w, arr_w.data, arr_w.data_size);
        
        void* buf_x = bo_x.map<void*>();
        std::memcpy(buf_x, arr_x.data, arr_x.data_size);

        // Sync to device
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::cout << "Executing kernel...\n";
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_w, bo_x, bo_y);
        run.wait();

        std::cout << "Kernel execution completed. Checking results...\n";
        bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        bf16* y_actual = bo_y.map<bf16*>();
        bf16* y_expected = static_cast<bf16*>(arr_expected.data);
        
        size_t n_elements = arr_expected.data_size / sizeof(bf16);
        bool passed = true;
        float max_diff = 0.0f;
        for (size_t i = 0; i < n_elements; i++) {
            float a = y_actual[i].to_float();
            float e = y_expected[i].to_float();
            float diff = std::abs(a - e);
            if (diff > max_diff) max_diff = diff;
            
            if (diff > 1.0f || std::isnan(a)) {
                std::cout << "Mismatch at " << i << ": expected " << e << ", actual " << a << "\n";
                passed = false;
                if (i > 10) break; // Don't spam output
            }
        }
        
        std::cout << "Max difference: " << max_diff << "\n";
        if (passed) {
            std::cout << "PASS: Single Matmul Parity vs Python matches!\n";
        } else {
            std::cout << "FAIL: Mismatch found.\n";
            return 1;
        }

        // Cleanup
        free_npy(arr_w);
        free_npy(arr_x);
        free_npy(arr_expected);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
