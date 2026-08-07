#include "alveare/server.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <chrono>
#include <csignal>
#include <thread>
#include <unistd.h>

using json = nlohmann::json;

namespace alveare {

static httplib::Server* g_active_server = nullptr;

static void handle_server_signal(int sig) {
    std::cout << "\n[alveare_runtime] Signal " << sig << " received. Shutting down server...\n" << std::flush;
    if (g_active_server) {
        g_active_server->stop();
    }
    // Launch a watchdog thread: if process teardown blocks on XRT context, force exit cleanly
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        std::cout << "[alveare_runtime] Shutdown watchdog: forcing exit now.\n" << std::flush;
        _exit(0);
    }).detach();
}

ApiServer::ApiServer(Generator& generator) : generator_(generator) {}

void ApiServer::stop() {
    if (svr_ptr_) {
        static_cast<httplib::Server*>(svr_ptr_)->stop();
    }
}

void ApiServer::start(int port) {
    httplib::Server svr;
    svr_ptr_ = &svr;
    g_active_server = &svr;

    std::signal(SIGINT, handle_server_signal);
    std::signal(SIGTERM, handle_server_signal);
    std::signal(SIGHUP, handle_server_signal);

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j_req = json::parse(req.body);
            std::string prompt = "";
            bool stream = false;
            bool enable_thinking = true;
            
            if (j_req.contains("stream") && j_req["stream"].is_boolean()) {
                stream = j_req["stream"].get<bool>();
            }

            if (j_req.contains("enable_thinking") && j_req["enable_thinking"].is_boolean()) {
                enable_thinking = j_req["enable_thinking"].get<bool>();
            }

            // Build the prompt. For Gemma we apply the model's chat template with
            // its special turn/channel tokens (the tokenizer matches them atomically);
            // other models just concatenate message contents.
            const std::string& model_type = generator_.config().model_type;
            bool is_gemma4 = generator_.config().is_gemma4();
            bool is_gemma3 = (model_type == "gemma3");

            if (j_req.contains("messages") && j_req["messages"].is_array()) {
                if (is_gemma3) {
                    // Gemma-3 uses the CLASSIC turn format (no channels, no thinking):
                    //   <bos><start_of_turn>role\n{content}<end_of_turn>\n<start_of_turn>model\n
                    // (Gemma-4's <|turn>/<|channel> tokens don't exist in Gemma-3, so
                    // applying that template split them into garbage -> word-salad output.)
                    prompt = "<bos>";
                    for (const auto& msg : j_req["messages"]) {
                        if (!msg.contains("content") || !msg["content"].is_string()) continue;
                        std::string role = msg.value("role", "user");
                        if (role == "assistant") role = "model";
                        prompt += "<start_of_turn>" + role + "\n";
                        prompt += msg["content"].get<std::string>() + "<end_of_turn>\n";
                    }
                    prompt += "<start_of_turn>model\n";
                } else if (is_gemma4) {
                    prompt = "<bos>";
                    // Per the Gemma-4 chat template: an EMPTY "<|channel>thought\n<channel|>"
                    // block after "<|turn>model\n" tells the model the thought is empty =>
                    // SKIP thinking. It is appended only when thinking is DISABLED. When
                    // thinking is ENABLED the generation prompt ends at "<|turn>model\n" and
                    // the model opens its own thought channel. (This was inverted before, so
                    // enabling the toggle actually suppressed thinking.)
                    std::string think_suppress = enable_thinking ? "" : "<|channel>thought\n<channel|>";
                    for (const auto& msg : j_req["messages"]) {
                        if (!msg.contains("content") || !msg["content"].is_string()) continue;
                        std::string role = msg.value("role", "user");
                        if (role == "assistant") role = "model";
                        // Completed turns are replayed as their content only; the ephemeral
                        // thought channel is not re-fed (the template strips it too).
                        prompt += "<|turn>" + role + "\n";
                        prompt += msg["content"].get<std::string>() + "<turn|>\n";
                    }
                    prompt += "<|turn>model\n" + think_suppress;
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
            // Sampling controls (OpenAI-style). temperature<=0 keeps greedy decoding.
            if (j_req.contains("temperature") && j_req["temperature"].is_number()) {
                params.temperature = j_req["temperature"].get<float>();
            }
            if (j_req.contains("top_p") && j_req["top_p"].is_number()) {
                params.top_p = j_req["top_p"].get<float>();
            }
            if (j_req.contains("top_k") && j_req["top_k"].is_number_integer()) {
                params.top_k = j_req["top_k"].get<int>();
            }
            if (j_req.contains("seed") && j_req["seed"].is_number_integer()) {
                params.seed = static_cast<unsigned>(j_req["seed"].get<long>());
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
    svr_ptr_ = nullptr;
    g_active_server = nullptr;
}

} // namespace alveare
