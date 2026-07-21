#include "alveare/server.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <chrono>

using json = nlohmann::json;

namespace alveare {

ApiServer::ApiServer(Generator& generator) : generator_(generator) {}

void ApiServer::start(int port) {
    httplib::Server svr;

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j_req = json::parse(req.body);
            std::string prompt = "";
            bool stream = false;
            
            if (j_req.contains("stream") && j_req["stream"].is_boolean()) {
                stream = j_req["stream"].get<bool>();
            }

            // Build the prompt. For Gemma we apply the model's chat template with
            // its special turn/channel tokens (the tokenizer matches them atomically);
            // other models just concatenate message contents.
            const std::string& model_type = generator_.config().model_type;
            bool is_gemma = (model_type == "gemma3" || model_type == "gemma4");

            if (j_req.contains("messages") && j_req["messages"].is_array()) {
                if (is_gemma) {
                    prompt = "<bos>";
                    for (const auto& msg : j_req["messages"]) {
                        if (!msg.contains("content") || !msg["content"].is_string()) continue;
                        std::string role = msg.value("role", "user");
                        if (role == "assistant") role = "model";
                        prompt += "<|turn>" + role + "\n";
                        // Replay a completed assistant turn WITH the same
                        // generation-prompt suffix the model saw when producing it,
                        // so its tokens match what is already in the KV cache and the
                        // whole conversation prefix is reused (no re-prefill).
                        if (role == "model") prompt += "<|channel>thought\n<channel|>";
                        prompt += msg["content"].get<std::string>() + "<turn|>\n";
                    }
                    prompt += "<|turn>model\n<|channel>thought\n<channel|>";
                } else {
                    for (const auto& msg : j_req["messages"]) {
                        if (msg.contains("content") && msg["content"].is_string()) {
                            prompt += msg["content"].get<std::string>() + "\n";
                        }
                    }
                }
            } else if (j_req.contains("prompt") && j_req["prompt"].is_string()) {
                prompt = j_req["prompt"].get<std::string>();
            }

            GenerationParams params;
            if (j_req.contains("max_tokens") && j_req["max_tokens"].is_number_integer()) {
                params.max_tokens = j_req["max_tokens"].get<int>();
            }

            std::string full_response = "";

            if (stream) {
                res.set_chunked_content_provider("text/event-stream",
                    [this, prompt, params](size_t offset, httplib::DataSink& sink) {
                        generator_.generate(prompt, params, [&](const std::string& token) {
                            json delta = {{"content", token}};
                            json chunk = {
                                {"id", "chatcmpl-123"},
                                {"object", "chat.completion.chunk"},
                                {"created", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                                {"model", "alveare-model"},
                                {"choices", {{
                                    {"index", 0},
                                    {"delta", delta},
                                    {"finish_reason", nullptr}
                                }}}
                            };
                            std::string sse = "data: " + chunk.dump() + "\n\n";
                            sink.write(sse.c_str(), sse.size());
                            return true; // continue
                        });
                        
                        // Done
                        std::string done_msg = "data: [DONE]\n\n";
                        sink.write(done_msg.c_str(), done_msg.size());
                        sink.done();
                        return true;
                    }
                );
            } else {
                generator_.generate(prompt, params, [&](const std::string& token) {
                    full_response += token;
                    return true;
                });
                
                json resp = {
                    {"id", "chatcmpl-123"},
                    {"object", "chat.completion"},
                    {"created", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                    {"model", "alveare-model"},
                    {"choices", {{
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", full_response}
                        }},
                        {"finish_reason", "stop"}
                    }}},
                    {"usage", {
                        {"prompt_tokens", 0},
                        {"completion_tokens", 0},
                        {"total_tokens", 0}
                    }}
                };

                res.set_content(resp.dump(), "application/json");
            }
        } catch (const std::exception& e) {
            json err = {{"error", {{"message", e.what()}}}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    std::cout << "Starting OpenAI compatible API server on port " << port << "...\n";
    svr.listen("0.0.0.0", port);
}

} // namespace alveare
