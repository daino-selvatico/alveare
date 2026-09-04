#include "alveare/model.h"
#include "alveare/weights.h"
#include "alveare/config.h"
#include "alveare/npu.h"
#include "alveare/bf16.h"
#include "alveare/npy.h"

#include "alveare/cpu_backend.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace alveare;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: layer_test <model_dir> [manifest.json] [--device npu|cpu]\n";
        return 1;
    }

    std::string model_dir = argv[1];
    std::string manifest_path = (argc >= 3 && argv[2][0] != '-') ? argv[2] : "";
    std::string device_str = "npu";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg.rfind("--device=", 0) == 0) {
            device_str = arg.substr(9);
        }
    }

    try {
        ModelConfig config = load_config(model_dir + "/config.json");
        std::unique_ptr<ComputeDevice> dev;
        if (device_str == "cpu") {
            std::cout << "Using CPU Compute Backend...\n";
            dev = std::make_unique<CpuBackend>();
        } else {
            std::cout << "Using NPU Compute Backend with manifest: " << manifest_path << "\n";
            dev = std::make_unique<NpuRegistry>(manifest_path);
        }

        ModelWeights mw = load_weights(model_dir, config, *dev);
        Model model(config, mw, *dev);

        int K = config.hidden_size;
        std::vector<bf16> x(K, bf16(0.01f)); // Dummy input
        std::vector<bf16> out(K, bf16(0.0f));

        std::cout << "Running layer 0...\n";
        model.run_layer(x.data(), 0, 0, out.data());
        
        std::cout << "Layer 0 output sample: " 
                  << out[0].to_float() << " " << out[1].to_float() << " ...\n";
                  
        std::cout << "SUCCESS!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
