#include "bpe.hpp"

#include <iostream>
#include <fstream>
#include <variant>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <queue>
#include <functional>
#include <random>
#include <stdexcept>
#include <locale>  // Potentially for std::locale fix (if needed)
#include <codecvt> // Potentially for std::wstring_convert (if needed)
#include "json.hpp"
#include "inja.hpp"

///////////////////////////////////////////////////////////////////////////////
//                               UTF-8 Helpers                               //
///////////////////////////////////////////////////////////////////////////////

/**
 * Convert a UTF-8 string into a vector of complete UTF-8 characters (codepoints).
 * Each element is a substring containing exactly one UTF-8 character.
 */
static std::vector<std::string> utf8_to_chars(const std::string &input)
{
    std::vector<std::string> chars;
    chars.reserve(input.size());

    for (size_t i = 0; i < input.size();)
    {
        unsigned char c = static_cast<unsigned char>(input[i]);
        int len;
        if ((c & 0xF8) == 0xF0)
        { // 4-byte UTF-8
            len = 4;
        }
        else if ((c & 0xF0) == 0xE0)
        { // 3-byte UTF-8
            len = 3;
        }
        else if ((c & 0xE0) == 0xC0)
        { // 2-byte UTF-8
            len = 2;
        }
        else
        { // 1-byte ASCII
            len = 1;
        }

        // Bounds check (avoid going past end):
        if (i + len > input.size())
        {
            len = 1; // fallback to 1 if truncated
        }
        chars.push_back(input.substr(i, len));
        i += len;
    }

    return chars;
}

/**
 * Replaces ASCII spaces ' ' with the 3-byte UTF-8 sequence for U+2581: "▁"
 * which is "\xE2\x96\x81" in UTF-8.
 */
static std::string replace_spaces_with_underline(const std::string &input, const std::string &special_character)
{
    // static const std::string UNDERLINE = "\xE2\x96\x81"; // "▁" in UTF-8
    std::string output;
    output.reserve(input.size() * 2); // just a guess to reduce reallocs
    for (char c : input)
    {
        if (c == ' ')
        {
            output += special_character;
        }
        else
        {
            output.push_back(c);
        }
    }
    return output;
}

///////////////////////////////////////////////////////////////////////////////
//                          Added Vocab (FastList)                           //
///////////////////////////////////////////////////////////////////////////////

struct Node
{
    std::string value;
    Node *prev;
    Node *next;
    explicit Node(const std::string &val) : value(val), prev(nullptr), next(nullptr) {}
};

class FastList
{
public:
    FastList() : head(nullptr), tail(nullptr) {}
    ~FastList() { clear(); }

    void append(const std::string &value)
    {
        Node *new_node = new Node(value);
        if (!head)
        {
            head = tail = new_node;
        }
        else
        {
            tail->next = new_node;
            new_node->prev = tail;
            tail = new_node;
        }
    }

    std::vector<std::string> to_vector() const
    {
        std::vector<std::string> result;
        for (Node *cur = head; cur != nullptr; cur = cur->next)
        {
            result.push_back(cur->value);
        }
        return result;
    }

    // Merges occurrences of word_list (already split into UTF-8 chars)
    // into single nodes.  If word_list == {"c","h","o","l","e","r","a"},
    // then whenever we see consecutive nodes c->h->o->l->...->a, we replace
    // them with a single node "cholera".
    void search_and_replace(const std::vector<std::string> &word_list)
    {
        if (word_list.size() < 2)
        {
            // For single-character or empty, we skip merges
            return;
        }
        Node *current = head;
        const int word_len = static_cast<int>(word_list.size());

        while (current)
        {
            Node *match_node = current;
            int i = 0;
            while (match_node && i < word_len && match_node->value == word_list[i])
            {
                match_node = match_node->next;
                ++i;
            }
            if (i == word_len)
            {
                // Full match found: replace [current, match_node) with single node
                replace_sequence(current, match_node, word_list);
                if (!current->prev)
                {
                    // replaced at head
                    current = head;
                }
                else
                {
                    // replaced in the middle
                    current = current->prev->next;
                }
                if (current)
                {
                    current = current->next;
                }
            }
            else
            {
                current = current->next;
            }
        }
    }

private:
    Node *head;
    Node *tail;

    void clear()
    {
        Node *cur = head;
        while (cur)
        {
            Node *tmp = cur;
            cur = cur->next;
            delete tmp;
        }
        head = tail = nullptr;
    }

    void replace_sequence(Node *start_node, Node *end_node,
                          const std::vector<std::string> &word_list)
    {
        // Join word_list into one string
        std::string merged;
        merged.reserve(8 * word_list.size()); // heuristic
        for (auto &piece : word_list)
        {
            merged += piece;
        }

        Node *new_node = new Node(merged);

        Node *prev_node = start_node->prev;
        Node *next_node = end_node; // end_node is NOT included in the match

        // Link in new_node
        if (prev_node)
        {
            prev_node->next = new_node;
            new_node->prev = prev_node;
        }
        else
        {
            head = new_node;
        }
        if (next_node)
        {
            next_node->prev = new_node;
            new_node->next = next_node;
        }
        else
        {
            tail = new_node;
        }

        // Free replaced nodes
        Node *cur = start_node;
        while (cur != end_node)
        {
            Node *tmp = cur;
            cur = cur->next;
            delete tmp;
        }
    }
};

/**
 * Merges each string in added_vocab (split into chars) into the token_list
 * in a greedy fashion.
 */
static std::vector<std::string> merge_added_vocab(
    const std::vector<std::string> &token_list,
    const std::vector<std::string> &added_vocab)
{
    if (added_vocab.empty())
    {
        return token_list;
    }

    // Sort descending by length => so the longest added vocab merges first
    std::vector<std::string> vocab_copy = added_vocab;
    std::sort(vocab_copy.begin(), vocab_copy.end(),
              [](const std::string &a, const std::string &b)
              {
                  return a.size() > b.size();
              });

    // Build a FastList from token_list
    FastList fl;
    for (auto &token : token_list)
    {
        fl.append(token);
    }

    // For each added_vocab word, convert to UTF-8 chars & search+replace
    for (auto &v_word : vocab_copy)
    {
        std::vector<std::string> chars = utf8_to_chars(v_word);
        fl.search_and_replace(chars);
    }

    return fl.to_vector();
}

///////////////////////////////////////////////////////////////////////////////
//                   Faster BPE (SentencePiece-style merges)                 //
///////////////////////////////////////////////////////////////////////////////

/**
 * A small function to detect if a piece ID is "unused" (out-of-vocab)
 * so that we can do resegmentation. Here, we do a trivial check:
 *    if (0 <= id < vocab_size) => it's "used"
 *    else => "unused"
 */
static bool is_unused_inlined(int id, int vocab_size)
{
    return (id < 0 || id >= vocab_size);
}

/**
 * The FasterBPE class is a priority-queue-based BPE engine.
 * It is used by the BPE class to perform the actual BPE merges.
 */
FasterBPE::FasterBPE(const std::map<std::pair<std::string, std::string>, int> &bpe_ranks,
                     const std::map<std::string, int> &vocab)
{
    // We want a quick lookup from "left+right" => rank
    // Also store vocab size for "unused" checks
    m_vocab_size = (int)vocab.size();

    for (auto &kv : bpe_ranks)
    {
        const auto &p = kv.first; // (left, right)
        int rank = kv.second;
        // key = left+right
        std::string concat = p.first + p.second;
        m_pieces[concat] = rank;
    }

    // For a quick piece->ID mapping
    for (auto &kv : vocab)
    {
        m_str2id[kv.first] = kv.second;
    }
}

/**
 * Takes a sequence of tokenized codepoints (after added_vocab merges)
 * and does the SentencePiece-style priority-queue BPE merges.
 *
 * - alpha is the BPE-dropout probability (0.0 => no dropout).
 * - Returns the final subword tokens (UTF-8).
 */
std::vector<std::string> FasterBPE::run_faster_bpe(const std::vector<std::string> &tokens,
                                                   float alpha) const
{
    if (tokens.empty())
    {
        return {};
    }

    // We define some internal data structures:
    struct Symbol
    {
        int prev;
        int next;
        bool freeze;
        std::string piece;
    };

    struct SymbolPair
    {
        int left;    // index in "symbols"
        int right;   // index in "symbols"
        float score; // bigger => higher priority
        size_t size; // total length of merged piece
    };

    // For the priority queue:
    struct SymbolPairComparator
    {
        bool operator()(const SymbolPair *a, const SymbolPair *b) const
        {
            // higher score => pop first
            if (a->score < b->score)
                return true;
            if (a->score > b->score)
                return false;
            // tie-break: smaller left index => higher priority
            return (a->left > b->left);
        }
    };

    // We also store "rev_merge" so we can resegment out-of-vocab pieces.
    std::unordered_map<std::string, std::pair<std::string, std::string>> rev_merge;

    // A quick function to retrieve a "score" from a rank => -rank
    auto get_score = [&](int rank)
    {
        // Lower rank => bigger score
        return (float)(-rank);
    };

    // A function to get the ID of a piece. -1 if not in m_str2id.
    auto piece_to_id = [&](const std::string &piece)
    {
        auto it = m_str2id.find(piece);
        if (it == m_str2id.end())
        {
            return -1;
        }
        return it->second;
    };

    // We want to maybe add a pair (left,right) if "left+right" is in merges
    auto MaybeAddNewSymbolPair = [&](int left_idx, int right_idx,
                                     auto &symbols, auto &agenda,
                                     auto &symbol_pair_alloc)
    {
        if (left_idx < 0 || right_idx < 0)
        {
            return;
        }
        if (symbols[left_idx].freeze || symbols[right_idx].freeze)
        {
            return;
        }
        const std::string &left_piece = symbols[left_idx].piece;
        const std::string &right_piece = symbols[right_idx].piece;
        if (left_piece.empty() || right_piece.empty())
        {
            return;
        }
        std::string merged = left_piece + right_piece;
        auto it = m_pieces.find(merged);
        if (it == m_pieces.end())
        {
            return; // not a known pair
        }
        int rank = it->second;
        SymbolPair *sp = symbol_pair_alloc();
        sp->left = left_idx;
        sp->right = right_idx;
        sp->score = get_score(rank);
        sp->size = merged.size();
        agenda.push(sp);

        // For re-segmentation: if piece is out-of-vocab, we store how to break it
        int pid = piece_to_id(merged);
        if (is_unused_inlined(pid, m_vocab_size))
        {
            rev_merge[merged] = {left_piece, right_piece};
        }
    };

    // 1) Convert 'tokens' into a linked list of Symbol
    std::vector<Symbol> symbols;
    symbols.reserve(tokens.size());

    {
        // We artificially re-split each token into codepoints if needed,
        // but typically each token is already a single or merged codepoint.
        // For demonstration, let's assume each string in 'tokens' is atomic.
        // We'll assign prev/next indices.
        // The key fix for multi-byte chars (like “▁”) is that they are
        // single, intact strings in 'tokens', not partial bytes.
        int idx = 0;
        for (auto &tk : tokens)
        {
            Symbol s;
            bool freeze_flag = false;
            // do a full UTF-8 prefix match => but we consider the entire tk at once
            // for demonstration, we won't actually “split” tk further.
            // In real usage, if tk has multiple codepoints, you might want to do that
            // only if needed. Or keep it as is if “added_vocab” already merged them.
            // int mblen = (int)tk.size();  // treat the entire token as 1 chunk
            // The essential part is: do NOT break multi-byte chars incorrectly
            // Here we skip the loop because 'tk' is presumably correct
            s.piece = tk;
            s.freeze = freeze_flag;
            s.prev = (idx == 0) ? -1 : (idx - 1);
            s.next = -1; // to fill below
            symbols.push_back(s);
            idx++;
        }
        // fix next pointers
        for (int i = 0; i < (int)symbols.size() - 1; i++)
        {
            symbols[i].next = i + 1;
        }
    }

    if (symbols.empty())
    {
        return {};
    }

    // 2) Build a priority queue of adjacent pairs
    using Agenda = std::priority_queue<SymbolPair *,
                                       std::vector<SymbolPair *>,
                                       SymbolPairComparator>;
    Agenda agenda;

    // We'll also keep a little factory for SymbolPair pointers
    // so we can manage them easily.
    struct SPFactory
    {
        std::vector<std::unique_ptr<SymbolPair>> pool;
        SymbolPair *operator()()
        {
            pool.emplace_back(new SymbolPair());
            return pool.back().get();
        }
    } symbol_pair_alloc;

    for (int i = 0; i + 1 < (int)symbols.size(); i++)
    {
        MaybeAddNewSymbolPair(i, i + 1, symbols, agenda, symbol_pair_alloc);
    }

    // 3) BPE-dropout logic
    auto skip_merge = [&](std::mt19937 &rng)
    {
        if (alpha <= 0.0f)
            return false;
        if (alpha >= 1.0f)
            return true;
        std::uniform_real_distribution<> dist(0.0, 1.0);
        return (dist(rng) < alpha);
    };
    std::random_device rd;
    std::mt19937 rng(rd());

    // 4) Repeatedly pop top pair, merge it, add new pairs
    while (!agenda.empty())
    {
        SymbolPair *top = agenda.top();
        agenda.pop();

        int L = top->left;
        int R = top->right;
        if (L < 0 || R < 0)
            continue;

        // Check if stale: maybe symbol was merged away
        if (symbols[L].piece.empty() || symbols[R].piece.empty())
        {
            continue;
        }
        size_t expected_len = symbols[L].piece.size() + symbols[R].piece.size();
        if (expected_len != top->size)
        {
            continue; // stale
        }
        // skip merge with probability alpha
        if (skip_merge(rng))
        {
            continue;
        }

        // Actually merge them
        symbols[L].piece += symbols[R].piece;
        symbols[R].piece.clear();

        // fix next/prev pointers
        int leftPrev = symbols[L].prev;
        int rightNext = symbols[R].next;

        symbols[L].next = rightNext;
        if (rightNext != -1)
        {
            symbols[rightNext].prev = L;
        }

        // Possibly re-add new pairs
        // (leftPrev, L) and (L, rightNext)
        if (leftPrev != -1)
        {
            MaybeAddNewSymbolPair(leftPrev, L, symbols, agenda, symbol_pair_alloc);
        }
        if (rightNext != -1)
        {
            MaybeAddNewSymbolPair(L, rightNext, symbols, agenda, symbol_pair_alloc);
        }
    }

    // 5) Re-segmentation for any out-of-vocab merges
    //    If a merged piece is "unused," we look up rev_merge to see how to break it.
    std::function<void(const std::string &, std::vector<std::string> &)> resegment;
    resegment = [&](const std::string &w, std::vector<std::string> &out)
    {
        // Check if it's in vocab
        int id = piece_to_id(w);
        if (id == -1 || is_unused_inlined(id, m_vocab_size))
        {
            // See if we can break it further
            auto it = rev_merge.find(w);
            if (it == rev_merge.end())
            {
                // No info => just keep it
                out.push_back(w);
                return;
            }
            // break it
            resegment(it->second.first, out);
            resegment(it->second.second, out);
        }
        else
        {
            // in vocab => done
            out.push_back(w);
        }
    };

    // Collect final pieces in order
    std::vector<std::string> result;
    result.reserve(symbols.size());

    // Start from index 0
    int cur_idx = 0;
    while (cur_idx != -1 && cur_idx < (int)symbols.size())
    {
        if (!symbols[cur_idx].piece.empty())
        {
            resegment(symbols[cur_idx].piece, result);
        }
        cur_idx = symbols[cur_idx].next;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
//                         BPE Wrapper Class                                 //
///////////////////////////////////////////////////////////////////////////////

BPE::BPE(
    const std::map<std::pair<std::string, std::string>, int> &bpe_ranks,
    const std::map<std::string, int> &vocab,
    const std::vector<std::string> &added_vocab,
    const std::string &special_character,
    const std::map<std::string, std::string> &token_replace_map,
    const std::map<std::string, std::string> &reverse_tokens_replace_map) : m_bpe_ranks(bpe_ranks),
                                                                            m_vocab(vocab),
                                                                            m_reverse_vocab(),
                                                                            m_added_vocab(added_vocab),
                                                                            m_special_character(special_character),
                                                                            m_token_replace_map(token_replace_map),
                                                                            m_reverse_tokens_replace_map(reverse_tokens_replace_map),
                                                                            m_faster_bpe(bpe_ranks, vocab)
{
    // Build the reverse vocabulary map during initialization
    for (const auto &[token, id] : m_vocab)
    {
        m_reverse_vocab[id] = token;
    }
}

std::string BPE::decode(const std::vector<int> &tokens)
{
    if (tokens.empty())
    {
        return "";
    }

    // Estimate initial capacity to avoid reallocations
    std::string result;
    result.reserve(tokens.size() * 8); // Reasonable estimate for token length

    const bool has_special_char = !m_special_character.empty();
    const size_t special_char_len = has_special_char ? m_special_character.length() : 0;
    for (const int id : tokens)
    {
        const auto it = m_reverse_vocab.find(id);
        if (it == m_reverse_vocab.end() || it->second.empty())
            continue;

        std::string token = it->second;

        // Check if the token is in the m_reverse_tokens_replace_map
        const auto replace_it = m_reverse_tokens_replace_map.find(token);
        if (replace_it != m_reverse_tokens_replace_map.end())
        {
            // Use the replacement value from the map
            token = replace_it->second;
        }

        if (!has_special_char)
        {
            // Fast path: no replacements needed
            result.append(token);
            continue;
        }

        // Process token with possible replacements
        size_t start = 0;
        size_t pos = token.find(m_special_character);

        if (pos == std::string::npos)
        {
            // No special character found
            result.append(token);
        }
        else
        {
            // Process all special character occurrences
            do
            {
                // Append segment before special character
                result.append(token, start, pos - start);
                // Add space instead of special character
                result.push_back(' ');
                // Move start position
                start = pos + special_char_len;
                // Find next occurrence
                pos = token.find(m_special_character, start);
            } while (pos != std::string::npos);

            // Append remaining part after last special character
            if (start < token.length())
            {
                result.append(token, start, token.length() - start);
            }
        }
    }

    return result;
}

// Main encode function:
//  1) Replace space -> "▁"
//  2) Split into full UTF-8 chars
//  3) Merge 'added_vocab'
//  4) Run faster BPE merges
//  5) Return final subwords
std::variant<std::vector<std::string>, std::vector<int>> BPE::encode(
    const std::string &text,
    float alpha,
    bool tokenize)
{
    // 1) Replace spaces with "▁"
    std::string replaced = replace_spaces_with_underline(text, m_special_character);

    // 2) Replace characters according to m_token_replace_map for consistency
    for (const auto &[original, replacement] : m_token_replace_map)
    {
        size_t pos = 0;
        while ((pos = replaced.find(original, pos)) != std::string::npos)
        {
            replaced.replace(pos, original.length(), replacement);
            pos += replacement.length();
        }
    }

    // 3) Convert to full UTF-8 chars
    std::vector<std::string> tokens = utf8_to_chars(replaced);

    // 4) Merge user-specified vocabulary first
    tokens = merge_added_vocab(tokens, m_added_vocab);

    // 5) Run the faster BPE merges (SentencePiece style)
    tokens = m_faster_bpe.run_faster_bpe(tokens, alpha);

    // 6) Convert tokens to token IDs if tokenize is true
    if (tokenize)
    {
        std::vector<int> token_ids;
        token_ids.reserve(tokens.size());

        for (const auto &token : tokens)
        {
            auto it = m_vocab.find(token);
            if (it != m_vocab.end())
            {
                token_ids.push_back(it->second);
            }
            else
            {
                // Handle unknown tokens with 0
                token_ids.push_back(0);
            }
        }

        return token_ids;
    }

    return tokens;
}