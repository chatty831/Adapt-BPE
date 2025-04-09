# Adapt-BPE

This repository provides tools for using AdaptBPE type tokenizers, as detailed in the [paper](https://arxiv.org/abs/2410.03258).

## Supported Tokenizers

- **SentencePiece Tokenizers:** Currently, the repository supports only SentencePiece tokenizers.
- **tiktoken Tokenizers:** Support for tiktoken tokenizers is planned for future releases.

## Installation

### Linux

Clone the repository and install it in editable mode:

```bash
pip install -e .
```

### Windows/macOS

For Windows and macOS users, you need to compile the repository before installation. To compile, run:

```bash
python setup.py build_ext --inplace
pip install -e .
```

**Note:** A C++17 (or later) compiler is required due to the usage of the `<variant>` header.

## How to Use

To instantiate an AdaptBPE tokenizer, you can use the following utility function as an example:

```python
def load_tokenizer(tokenizer_path: str, added_vocab=[]):
    # Mapping of characters to their token representations.
    # This replace map is typically derived from the tokenizer's configuration file.
    # Note: For many tokenizers, the replace map is variable. To get the correct mapping,
    # check the tokenizer.json file in your tokenizer directory for tokens with hex character keys
    # and update the map to replace them with their actual representation.
    llama_2_token_replace_map = {
        '\n': '<0x0A>',  
        '\t': '<0x09>',  
        '\b': '<0x08>',  
        '\f': '<0x0C>',  
        '\a': '<0x07>',  
        '\v': '<0x0B>',  
        "\'": "'",
        '\x00': '<0x00>',
        '\x01': '<0x01>',
        '\x02': '<0x02>',
        '\x03': '<0x03>',
        '\x04': '<0x04>',
        '\x05': '<0x05>',
        '\x06': '<0x06>',
        '\x07': '<0x07>',
        '\x08': '<0x08>',
        '\x0b': '<0x0B>',
        '\x0c': '<0x0C>',
        '\x0e': '<0x0E>',
        '\x0f': '<0x0F>',
        '\x10': '<0x10>',
        '\x11': '<0x11>',
        '\x12': '<0x12>',
        '\x13': '<0x13>',
        '\x14': '<0x14>',
        '\x15': '<0x15>',
        '\x16': '<0x16>',
        '\x17': '<0x17>',
        '\x18': '<0x18>',
        '\x19': '<0x19>',
        '\x1a': '<0x1A>',
        '\x1b': '<0x1B>',
        '\x1c': '<0x1C>',
        '\x1d': '<0x1D>',
        '\x1e': '<0x1E>',
        '\x1f': '<0x1F>',
        '\x7f': '<0x7F>',
        '\x84': '<0x84>',
        '\x85': '<0x85>',
        '\x86': '<0x86>',
        '\x87': '<0x87>',
        '\x88': '<0x88>',
        '\x89': '<0x89>',
        '\x8a': '<0x8A>',
        '\x8b': '<0x8B>',
        '\x8c': '<0x8C>',
        '\x8d': '<0x8D>',
        '\x8e': '<0x8E>',
        '\x8f': '<0x8F>',
        '\x90': '<0x90>',
        '\x95': '<0x95>',
        '\x96': '<0x96>',
        '\x98': '<0x98>',
        '\x9a': '<0x9A>',
        '\x9b': '<0x9B>',
        '\x9e': '<0x9E>',
        '\x9f': '<0x9F>',       
    }
    
    """
    Load an AdaptBPE tokenizer with support for adapted BPE.
    
    Parameters:
      tokenizer_path : str
          The file path to the tokenizer model.
      added_vocab : list, optional
          A list of additional vocabulary tokens to include (default is an empty list).
    
    Keyword Arguments:
      special_character : str
          A marker used to denote token boundaries (commonly "▁").
      token_replace_map : dict
          Dictionary mapping tokens with hex character keys (as found in tokenizer.json)
          to their corresponding actual representation.
          Note: This mapping may vary between tokenizers. To determine the correct mapping,
          inspect the tokenizer.json file in your tokenizer directory.
    
    Returns:
      An instance of AdaptBPETokenizer.
    """
    tokenizer = AdaptBPETokenizer(
        model_path=tokenizer_path, 
        # added_vocab=added_vocab,  # Uncomment to include additional vocabulary tokens.
        special_character="▁",
        token_replace_map=llama_2_token_replace_map,
    )
    return tokenizer

# Example usage:
# tokenizer = load_tokenizer("/path/to/your/tokenizer")
```