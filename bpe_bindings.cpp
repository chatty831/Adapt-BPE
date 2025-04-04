#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic conversion of STL types (e.g. std::vector<string>)
#include "bpe.hpp"        // This is where your BPE are defined.
#include "inja.hpp"
#include "json.hpp"
#include <string>
#include <regex>
#include <sstream>
#include <vector>
#include <map>

using json = nlohmann::json;

// Simple string trim function to replace the Jinja "trim" filter
std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    
    auto end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

// Custom mini template engine that handles the specific template patterns we need
std::string render_template(
    const std::string& tmpl,
    const std::vector<std::map<std::string, std::string>>& messages,
    const std::map<std::string, std::string>& special_tokens) {
    
    std::stringstream result;
    
    // Handle bos_token lookup
    std::string bos_token = "";
    auto it = special_tokens.find("bos_token");
    if (it != special_tokens.end()) {
        bos_token = it->second;
    }
    
    // Process each message in the conversation
    bool first_message = true;
    for (const auto& message : messages) {
        std::string role;
        std::string content;
        
        // Extract role and content
        auto role_it = message.find("role");
        if (role_it != message.end()) {
            role = role_it->second;
        }
        
        auto content_it = message.find("content");
        if (content_it != message.end()) {
            content = trim(content_it->second);
        }
        
        // Construct the formatted message
        std::string formatted = "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + "<|eot_id|>";
        
        // Add bos_token for the first message if available
        if (first_message && !bos_token.empty()) {
            formatted = bos_token + formatted;
        }
        
        result << formatted;
        first_message = false;
    }
    
    // Handle generation prompt if specified
    // This is a simple check for add_generation_prompt in the template
    if (tmpl.find("add_generation_prompt") != std::string::npos) {
        // Check if we should add the generation prompt by looking at special tokens
        bool add_generation_prompt = false;
        auto gen_it = special_tokens.find("add_generation_prompt");
        if (gen_it != special_tokens.end() && gen_it->second == "true") {
            add_generation_prompt = true;
        }
        
        if (add_generation_prompt) {
            result << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    }
    
    return result.str();
}

std::string apply_chat_template(
    const std::vector<std::map<std::string, std::string>>& conversation,
    const std::string& chat_template,
    const std::map<std::string, std::string>& special_tokens_map = {}) {
    
    // Check if chat template exists
    if (chat_template.empty()) {
        throw std::runtime_error("No chat template found in the tokenizer.");
    }
    
    // Check if template appears to be a valid Jinja-like template
    if (chat_template.find("{{") == std::string::npos && 
        chat_template.find("{%") == std::string::npos) {
        throw std::runtime_error("The chat_template doesn't appear to be a valid template.");
    }
    
    try {
        // Use our custom renderer instead of Inja
        return render_template(chat_template, conversation, special_tokens_map);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to render chat template: " + std::string(e.what()));
    }
}

namespace py = pybind11;

PYBIND11_MODULE(bpe_module, m)
{
    m.doc() = "Pybind11 wrapper for Faster BPE-like tokenizer";
    m.def("apply_chat_template", &apply_chat_template, 
          py::arg("conversation"), 
          py::arg("chat_template"), 
          py::arg("special_tokens_map") = std::map<std::string, std::string>(),
          "Apply a Jinja-like chat template to a conversation data structure");

    // Now we wrap the BPE class. We'll expose the constructor and the encode method.
    py::class_<BPE>(m, "BPE")
        // Expose constructor. We pass in references to the needed data structures.
        .def(py::init<const std::map<std::pair<std::string, std::string>, int>&,
                      const std::map<std::string, int>&,
                      const std::vector<std::string>&,
                      const std::string&,
                      const std::map<std::string, std::string>&,
                      const std::map<std::string, std::string>&>(), // Added reverse_tokens_replace_map parameter
             py::arg("bpe_ranks"),
             py::arg("vocab"),
             py::arg("added_vocab") = std::vector<std::string>(),
             py::arg("special_character") = "\xE2\x96\x81",
             py::arg("token_replace_map") = std::map<std::string, std::string>(),
             py::arg("reverse_tokens_replace_map") = std::map<std::string, std::string>()
        )

        // Expose the encode method
        .def("encode",
             &BPE::encode,
             "Encode a string using BPE",
             py::arg("text"),
             py::arg("alpha") = 0.0f,
             py::arg("tokenize") = true
        )

        // Expose the decode method
        .def("decode",
             &BPE::decode,
             "Decode a list of token IDs back into strings",
             py::arg("tokens")
        );
}
