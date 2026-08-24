#include "alveare/server.h"
#include "alveare/vision_embedder.h"
#include "alveare/audio_embedder.h"
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

ApiServer::ApiServer(Generator& generator, VisionEmbedder* vision_embedder, AudioEmbedder* audio_embedder)
    : generator_(generator), vision_embedder_(vision_embedder), audio_embedder_(audio_embedder) {}

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
            bool enable_thinking = (model_type != "gemma4-e4b" && model_type != "gemma4-e2b");
            
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

            std::vector<std::vector<float>> all_visual_embeddings;
            std::vector<std::vector<float>> all_audio_embeddings;

            auto process_image = [&](const std::string& url_or_b64) -> std::string {
                if (!vision_embedder_ || !vision_embedder_->is_loaded() || url_or_b64.empty()) {
                    return "[Immagine Allegata]";
                }
                auto emb = vision_embedder_->encode_image_base64(url_or_b64);
                if (!emb.empty()) {
                    std::cout << "[server] Successfully encoded image into " << emb.size() 
                              << " visual tokens (dim=" << (emb.empty() ? 0 : emb[0].size()) << ")\n" << std::flush;
                    std::string img_tag = "<|image>";
                    for (size_t i = 0; i < emb.size(); ++i) img_tag += "<|image|>";
                    img_tag += "<image|>";
                    all_visual_embeddings.insert(all_visual_embeddings.end(), emb.begin(), emb.end());
                    return img_tag;
                }
                std::cerr << "[server] Failed to decode image payload\n" << std::flush;
                return "[Immagine Allegata]";
            };

            auto process_audio = [&](const std::string& url_or_b64) -> std::string {
                if (!audio_embedder_ || !audio_embedder_->is_loaded() || url_or_b64.empty()) {
                    return "[Audio Allegato]";
                }
                auto emb = audio_embedder_->encode_audio_base64(url_or_b64);
                if (!emb.empty()) {
                    std::cout << "[server] Successfully encoded audio into " << emb.size() 
                              << " audio tokens (dim=" << (emb.empty() ? 0 : emb[0].size()) << ")\n" << std::flush;
                    std::string aud_tag = "<|audio>";
                    for (size_t i = 0; i < emb.size(); ++i) aud_tag += "<|audio|>";
                    aud_tag += "<audio|>";
                    all_audio_embeddings.insert(all_audio_embeddings.end(), emb.begin(), emb.end());
                    return aud_tag;
                }
                std::cerr << "[server] Failed to decode audio payload\n" << std::flush;
                return "[Audio Allegato]";
            };

            auto get_msg_content = [&](const nlohmann::json& msg) -> std::string {
                std::string visual_tags = "";
                std::string audio_tags = "";
                std::string text_body = "";

                // Check for root-level "images" array
                if (msg.contains("images") && msg["images"].is_array()) {
                    for (const auto& img_item : msg["images"]) {
                        if (img_item.is_string()) {
                            std::string tag = process_image(img_item.get<std::string>());
                            if (!visual_tags.empty()) visual_tags += "\n";
                            visual_tags += tag;
                        }
                    }
                }
                if (msg.contains("image_url") && msg["image_url"].is_string()) {
                    std::string tag = process_image(msg["image_url"].get<std::string>());
                    if (!visual_tags.empty()) visual_tags += "\n";
                    visual_tags += tag;
                }

                // Check for root-level "audio" / "input_audio"
                if (msg.contains("audio") && msg["audio"].is_string()) {
                    std::string tag = process_audio(msg["audio"].get<std::string>());
                    if (!audio_tags.empty()) audio_tags += "\n";
                    audio_tags += tag;
                }
                if (msg.contains("input_audio")) {
                    if (msg["input_audio"].is_string()) {
                        std::string tag = process_audio(msg["input_audio"].get<std::string>());
                        if (!audio_tags.empty()) audio_tags += "\n";
                        audio_tags += tag;
                    } else if (msg["input_audio"].is_object() && msg["input_audio"].contains("data")) {
                        std::string tag = process_audio(msg["input_audio"]["data"].get<std::string>());
                        if (!audio_tags.empty()) audio_tags += "\n";
                        audio_tags += tag;
                    }
                }

                if (msg.contains("content")) {
                    if (msg["content"].is_string()) {
                        std::string text = msg["content"].get<std::string>();
                        if (!text.empty()) {
                            if (!text_body.empty()) text_body += "\n";
                            text_body += text;
                        }
                    } else if (msg["content"].is_array()) {
                        for (const auto& part : msg["content"]) {
                            if (part.is_object()) {
                                std::string ptype = part.value("type", "");
                                if ((ptype == "text" || ptype.empty()) && part.contains("text")) {
                                    if (!text_body.empty()) text_body += "\n";
                                    text_body += part["text"].get<std::string>();
                                } else if (ptype == "image_url" || ptype == "image") {
                                    std::string url = "";
                                    if (part.contains("image_url")) {
                                        if (part["image_url"].is_string()) url = part["image_url"].get<std::string>();
                                        else if (part["image_url"].is_object() && part["image_url"].contains("url")) url = part["image_url"]["url"].get<std::string>();
                                    } else if (part.contains("image") && part["image"].is_string()) {
                                        url = part["image"].get<std::string>();
                                    } else if (part.contains("url") && part["url"].is_string()) {
                                        url = part["url"].get<std::string>();
                                    }
                                    std::string tag = process_image(url);
                                    if (!visual_tags.empty()) visual_tags += "\n";
                                    visual_tags += tag;
                                } else if (ptype == "input_audio" || ptype == "audio_url" || ptype == "audio") {
                                    std::string aud_data = "";
                                    if (part.contains("input_audio")) {
                                        if (part["input_audio"].is_string()) aud_data = part["input_audio"].get<std::string>();
                                        else if (part["input_audio"].is_object() && part["input_audio"].contains("data")) aud_data = part["input_audio"]["data"].get<std::string>();
                                    } else if (part.contains("audio_url")) {
                                        if (part["audio_url"].is_string()) aud_data = part["audio_url"].get<std::string>();
                                        else if (part["audio_url"].is_object() && part["audio_url"].contains("url")) aud_data = part["audio_url"]["url"].get<std::string>();
                                    } else if (part.contains("data") && part["data"].is_string()) {
                                        aud_data = part["data"].get<std::string>();
                                    } else if (part.contains("url") && part["url"].is_string()) {
                                        aud_data = part["url"].get<std::string>();
                                    }
                                    std::string tag = process_audio(aud_data);
                                    if (!audio_tags.empty()) audio_tags += "\n";
                                    audio_tags += tag;
                                }
                            }
                        }
                    }
                }

                std::string header_tags = "";
                if (!visual_tags.empty()) header_tags += visual_tags;
                if (!audio_tags.empty()) {
                    if (!header_tags.empty()) header_tags += "\n";
                    header_tags += audio_tags;
                }

                if (!header_tags.empty() && !text_body.empty()) {
                    return header_tags + "\n" + text_body;
                }
                return !header_tags.empty() ? header_tags : text_body;
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
                    } else if (model_type == "gemma4-e4b" || model_type == "gemma4-e2b") {
                        prompt += "<|turn>model\n";
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

            std::cout << "[server] Request received: " << all_visual_embeddings.size() 
                      << " visual tokens, " << all_audio_embeddings.size()
                      << " audio tokens. Prompt length: " << prompt.length() << " chars\n" << std::flush;

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
                    [this, prompt, params, model_name, is_gemma4, enable_thinking, all_visual_embeddings, all_audio_embeddings](size_t offset, httplib::DataSink& sink) {
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
                        }, all_visual_embeddings, all_audio_embeddings);

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
                }, all_visual_embeddings, all_audio_embeddings);
                
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

