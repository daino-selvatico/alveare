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

static inline void append_escaped_json(std::string& out, const std::string& s) {
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
}

void ApiServer::start(int port) {
    httplib::Server svr;
    svr_ptr_ = &svr;
    g_active_server = &svr;

    std::signal(SIGINT, handle_server_signal);
    std::signal(SIGTERM, handle_server_signal);
    std::signal(SIGHUP, handle_server_signal);

    const std::string& model_type = generator_.config().model_type;
    const std::string model_name = model_type.empty() ? "alveare-model" : model_type;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&, model_name](const httplib::Request&, httplib::Response& res) {
        json models_resp = {
            {"object", "list"},
            {"data", {{
                {"id", model_name},
                {"object", "model"},
                {"created", 1700000000},
                {"owned_by", "alveare"}
            }}}
        };
        res.set_content(models_resp.dump(), "application/json");
    });

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
            bool is_gemma4 = generator_.config().is_gemma4();
            bool is_gemma3 = (model_type == "gemma3");

            auto get_msg_content = [](const nlohmann::json& msg) -> std::string {
                if (!msg.contains("content")) return "";
                if (msg["content"].is_string()) return msg["content"].get<std::string>();
                if (msg["content"].is_array()) {
                    std::string res = "";
                    for (const auto& part : msg["content"]) {
                        if (part.is_object()) {
                            if (part.value("type", "") == "text" && part.contains("text")) {
                                if (!res.empty()) res += "\n";
                                res += part["text"].get<std::string>();
                            } else if (part.value("type", "") == "image_url") {
                                if (!res.empty()) res += "\n";
                                res += "[Immagine Allegata]";
                            } else if (part.value("type", "") == "input_audio") {
                                if (!res.empty()) res += "\n";
                                res += "[Audio Allegato]";
                            }
                        }
                    }
                    return res;
                }
                return "";
            };

            if (j_req.contains("messages") && j_req["messages"].is_array()) {
                if (is_gemma3) {
                    prompt = "<bos>";
                    for (const auto& msg : j_req["messages"]) {
                        std::string c = get_msg_content(msg);
                        if (c.empty()) continue;
                        std::string role = msg.value("role", "user");
                        if (role == "assistant") role = "model";
                        prompt += "<start_of_turn>" + role + "\n";
                        prompt += c + "<end_of_turn>\n";
                    }
                    prompt += "<start_of_turn>model\n";
                } else if (is_gemma4) {
                    prompt = "<bos>";
                    for (const auto& msg : j_req["messages"]) {
                        std::string c = get_msg_content(msg);
                        if (c.empty()) continue;
                        std::string role = msg.value("role", "user");
                        if (role == "assistant") role = "model";
                        prompt += "<|turn>" + role + "\n";
                        prompt += c + "<turn|>\n";
                    }
                    if (enable_thinking) {
                        prompt += "<|turn>model\n<|channel>thought\n";
                    } else {
                        prompt += "<|turn>model\n<|channel>thought\n<channel|>";
                    }
                } else {
                    for (const auto& msg : j_req["messages"]) {
                        std::string c = get_msg_content(msg);
                        if (!c.empty()) {
                            prompt += c + "\n";
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
            if (j_req.contains("stop")) {
                if (j_req["stop"].is_string()) {
                    params.stop.push_back(j_req["stop"].get<std::string>());
                } else if (j_req["stop"].is_array()) {
                    for (const auto& s : j_req["stop"]) {
                        if (s.is_string()) params.stop.push_back(s.get<std::string>());
                    }
                }
            }

            std::string full_response = "";

            if (stream) {
                res.set_chunked_content_provider("text/event-stream",
                    [this, prompt, params, model_name, is_gemma4, enable_thinking](size_t offset, httplib::DataSink& sink) {
                        auto req_id = "chatcmpl-" + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
                        int64_t created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                        std::string prefix = "data: {\"id\":\"" + req_id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created) + ",\"model\":\"" + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";
                        std::string suffix = "\"},\"finish_reason\":null}]}\n\n";

                        std::string sse_buf;
                        sse_buf.reserve(512);

                        if (is_gemma4 && enable_thinking) {
                            sse_buf = prefix;
                            append_escaped_json(sse_buf, "<|channel>thought\n");
                            sse_buf += suffix;
                            sink.write(sse_buf.c_str(), sse_buf.size());
                        }

                        generator_.generate(prompt, params, [&](const std::string& token) {
                            sse_buf.clear();
                            sse_buf += prefix;
                            append_escaped_json(sse_buf, token);
                            sse_buf += suffix;
                            sink.write(sse_buf.c_str(), sse_buf.size());
                            return true; // continue
                        });

                        std::string stop_chunk = "data: {\"id\":\"" + req_id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created) + ",\"model\":\"" + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
                        sink.write(stop_chunk.c_str(), stop_chunk.size());

                        std::string done_msg = "data: [DONE]\n\n";
                        sink.write(done_msg.c_str(), done_msg.size());
                        sink.done();
                        return true;
                    }
                );
            } else {
                if (is_gemma4 && enable_thinking) {
                    full_response = "<|channel>thought\n";
                }
                GenerationStats stats = generator_.generate(prompt, params, [&](const std::string& token) {
                    full_response += token;
                    return true;
                });
                
                json resp = {
                    {"id", "chatcmpl-" + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count())},
                    {"object", "chat.completion"},
                    {"created", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
                    {"model", model_name},
                    {"choices", {{
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", full_response}
                        }},
                        {"finish_reason", "stop"}
                    }}},
                    {"usage", {
                        {"prompt_tokens", stats.prompt_tokens},
                        {"completion_tokens", stats.completion_tokens},
                        {"total_tokens", stats.prompt_tokens + stats.completion_tokens}
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

