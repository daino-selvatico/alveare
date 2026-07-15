#include "alveare/model.h"
#include "alveare/weights.h"
#include "alveare/config.h"
#include "alveare/npu.h"
#include "alveare/bf16.h"
#include "alveare/npy.h"

#include <iostream>
#include <vector>
#include <string>

using namespace alveare;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: layer_test <model_dir> <manifest.json>\n";
        return 1;
    }

    std::string model_dir = argv[1];
    std::string manifest_path = argv[2];

    try {
        ModelConfig config = load_config(model_dir + "/config.json");
        NpuRegistry reg(manifest_path);

        ModelWeights mw = load_weights(model_dir, config, reg);
        Model model(config, mw, reg);

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
