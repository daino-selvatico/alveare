#include <iostream>
#include <string>
#include <memory>
#include "alveare/config.h"
#include "alveare/weights.h"
#include "alveare/npu.h"
#include "alveare/model.h"
#include "alveare/tokenizer.h"
#include "alveare/generator.h"
#include "alveare/server.h"

using namespace alveare;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: alveare_runtime <model_dir> <manifest.json> [port]\n";
        return 1;
    }

    std::string model_dir = argv[1];
    std::string manifest_path = argv[2];
    int port = 8080;
    if (argc >= 4) {
        port = std::stoi(argv[3]);
    }

    try {
        std::cout << "Loading config from " << model_dir << "/config.json\n";
        ModelConfig config = load_config(model_dir + "/config.json");
        
        std::cout << "Initializing NPU Registry with manifest: " << manifest_path << "\n";
        NpuRegistry reg(manifest_path);

        std::cout << "Loading model weights...\n";
        ModelWeights mw = load_weights(model_dir, config, reg);
        
        Model model(config, mw, reg);

        std::unique_ptr<Tokenizer> tokenizer;
        std::string tok_path = model_dir + "/tokenizer.json";
        try {
            tokenizer = std::make_unique<GemmaTokenizer>(tok_path);
            std::cout << "Loaded tokenizer from " << tok_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: no usable tokenizer (" << e.what()
                      << "), falling back to byte StubTokenizer.\n";
            tokenizer = std::make_unique<StubTokenizer>();
        }

        Generator generator(model, mw, *tokenizer);
        std::cout << "Model ready.\n";

        ApiServer server(generator);
        server.start(port);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
