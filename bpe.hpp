#ifndef BPE_HPP
#define BPE_HPP

#include <map>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>

/**
 * Internal BPE engine used by the wrapper class below.
 *
 * Declaration moved from bpe.cpp into bpe.hpp so that
 * BPE can store a `FasterBPE` object by value.
 */
class FasterBPE
{
public:
    FasterBPE(const std::map<std::pair<std::string, std::string>, int> &bpe_ranks,
              const std::map<std::string, int> &vocab);

    std::vector<std::string> run_faster_bpe(const std::vector<std::string> &tokens,
                                            float alpha = 0.0f) const;

private:
    std::unordered_map<std::string, int> m_pieces; // "left+right" => rank
    std::unordered_map<std::string, int> m_str2id; // piece => ID
    int m_vocab_size;
};

/**
 * The high-level BPE wrapper (main class).
 */
class BPE
{
public:
    BPE(
        const std::map<std::pair<std::string, std::string>, int> &bpe_ranks,
        const std::map<std::string, int> &vocab,
        const std::vector<std::string> &added_vocab = {},
        const std::string &special_character = "\xE2\x96\x81",
        const std::map<std::string, std::string> &token_replace_map = {},
        const std::map<std::string, std::string> &reverse_tokens_replace_map = {}
    );

    std::variant<std::vector<std::string>, std::vector<int>> encode(
        const std::string &text,
        float alpha = 0.0f,
        bool tokenize = true);

    std::string decode(
        const std::vector<int> &tokens);

private:
    std::map<std::pair<std::string, std::string>, int> m_bpe_ranks;
    std::map<std::string, int> m_vocab;
    std::map<int, std::string> m_reverse_vocab; // Added reverse vocabulary map
    std::vector<std::string> m_added_vocab;
    std::string m_special_character;
    std::map<std::string, std::string> m_token_replace_map;
    std::map<std::string, std::string> m_reverse_tokens_replace_map;
    
    FasterBPE m_faster_bpe; // composition of the FasterBPE engine
};

#endif