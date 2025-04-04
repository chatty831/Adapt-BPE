import json
import os
from typing import List, Dict, Any, Optional, Union, Literal
from jinja2 import Template
import torch
from torch import tensor, Tensor
import bpe_module
import re
from bpe_module import BPE


class AdaptBPETokenizer:
    def __init__(self, model_path, special_character, token_replace_map=None, added_vocab=[]):
        self.model_path = model_path
        self.special_character = special_character
        self.tokens_replace_map = token_replace_map
        self.reverse_token_replace_map = {v: k for k, v in token_replace_map.items()} if token_replace_map else {}
        self.config_path = os.path.join(self.model_path, "tokenizer_config.json")
        
        if not os.path.exists(self.config_path):
            raise RuntimeError(f"Configuration file not found at: {self.config_path}")
        
        self.tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
        
        if not os.path.exists(self.tokenizer_path):
            raise RuntimeError(f"Tokenizer file not found at: {self.tokenizer_path}")
        
        self.added_tokens_path = os.path.join(self.model_path, "added_vocab.txt")
        
        if not os.path.exists(self.added_tokens_path):
            if os.path.exists(os.path.join(self.model_path, "added_vocab.json")):
                self.added_tokens_path = os.path.join(self.model_path, "added_vocab.json")
            else:
                pass
        
        self.load_config()
        self.load_tokenizer()
        self.load_added_tokens()

        self.add_special_tokens(added_vocab)

        self.bpe_processor = BPE(
            bpe_ranks=self.bpe_ranks,
            vocab=self.vocab,
            added_vocab=self.added_vocab, 
            special_character=self.special_character,
            token_replace_map=token_replace_map,
            reverse_tokens_replace_map=self.reverse_token_replace_map,
        )
    
    
    def __call__(self, text: str, return_tensors: Literal["pt", "none"]="none", device=None, **kwargs) -> Union[List[int], Tensor]:
        if return_tensors=='pt':
            input_ids = tensor(self.encode(text, add_special_tokens=False, tokenize=True)).unsqueeze(0)  # Add extra shape for batch = 1
            attention_mask = tensor([1] * len(input_ids[0])).unsqueeze(0)  # Add extra shape for batch = 1
            if device is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        elif return_tensors=='none':
            input_ids = self.encode(text, add_special_tokens=False, tokenize=True)
            attention_mask = [1] * len(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
    
    def decode(self, ids: Union[List[int], Tensor]) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return self.bpe_processor.decode(tokens=ids) 
    
    def encode(self, text: str, tokenize=True, add_special_tokens=True) -> List[int]:
        if add_special_tokens:
            text = f"{self.bos_token}{text}"
        return self.bpe_processor.encode(text, tokenize=tokenize)
    
    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        python=True,
    ) -> str:
        
        if not hasattr(self, "chat_template") or self.chat_template is None:
            raise ValueError("No chat template found in the tokenizer.")
            
        if not re.search(r"(\{\{|\{%|\{#)", self.chat_template):
            raise ValueError("The chat_template doesn't appear to be a valid Jinja template.")
        
        try:
            if python:
                template = Template(self.chat_template)
                rendered_chat = template.render(
                    messages=conversation,
                    **self.special_tokens_map if hasattr(self, "special_tokens_map") else {}
                )
                return rendered_chat
            
            return bpe_module.apply_chat_template(
                conversation=conversation,
                chat_template=self.chat_template,
                special_tokens_map=self.special_tokens_map,
            )
        except Exception as e:
            raise ValueError(f"Failed to render chat template: {str(e)}")
    
    
    def add_special_tokens(self, added_vocab=[]):
        for token in self.special_tokens:
            self.vocab[token] = self.special_tokens[token]
            self.added_vocab.append(token)
        self.added_vocab.extend(added_vocab)
        
        for k,v in self.tokens_replace_map.items():
            self.added_vocab.append(v)
            
    def __repr__(self):
        return (f"AdaptBPETokenizer(\n\tmodel_path= {self.model_path},\n\t"
                f"special_character= {self.special_character},\n\t"
                f"added_tokens_count= {len(self.added_vocab)},\n\t"
                f"vocab_size= {len(self.vocab)},\n\t"
                f"bpe_ranks_count= {len(self.bpe_ranks)},\n\t"
                f"added_tokens_path= {self.added_tokens_path},\n\t"
                f"tokenizer_path= {self.tokenizer_path}\n)")
    
    def load_added_tokens(self):
        self.added_vocab = []
        if not os.path.exists(self.added_tokens_path):
            return
        
        with open(self.added_tokens_path, "r") as f:
            self.added_vocab = [line.strip() for line in f if line.strip()]
        
    def load_tokenizer(self):
        with open(self.tokenizer_path, "r") as f:
            tokenizer = json.load(f)
        
        added_tokens: List[Dict[str, Any]] = tokenizer.get("added_tokens", [])
        for added_token in added_tokens:
            if added_token.get("special", False):
                self.special_tokens[added_token['content']] = int(added_token['id'])
            else:
                self.vocab[added_token['content']] = int(added_token['id'])
        
        vocab = tokenizer['model']['vocab']
        for k, v in vocab.items():
            self.vocab[k] = int(v)
        
        self.bpe_ranks = {}
        merges: Union[List[str], List[List[str]]] = tokenizer['model']['merges']
        for i, merge in enumerate(merges):
            if isinstance(merge, list):
                self.bpe_ranks[tuple(merge)] = i
            else:
                self.bpe_ranks[tuple(merge.split(" "))] = i
        
        # print(self.bpe_ranks)
        
    def load_config(self):
        with open(self.config_path, "r") as f:
            config: Dict[str, Any] = json.load(f)
        
        added_tokens_decoder: Dict[str, Dict[str, Any]] = config.get("added_tokens_decoder", {})
        if not added_tokens_decoder:
            return
        special_tokens = {}
        tokens = {}
        for key, value in added_tokens_decoder.items():
            if value.get("special"):
                special_tokens[value['content']] = int(key)
            else:
                tokens[value['content']] = int(key)
        
        self.chat_template = config.get("chat_template", "")
        self.bos_token = config.get("bos_token", "")
        self.eos_token = config.get("eos_token", "")
        self.pad_token = config.get("pad_token", "")
        self.unk_token = config.get("unk_token", "")
        
        self.bos_token_id = special_tokens.get(self.bos_token, None) if self.bos_token else None
        self.eos_token_id = special_tokens.get(self.eos_token, None) if self.eos_token else None
        self.pad_token_id = special_tokens.get(self.pad_token, None) if self.pad_token else None
        self.unk_token_id = special_tokens.get(self.unk_token, None) if self.unk_token else None
        
        self.special_tokens = special_tokens
        self.vocab = tokens
        
        self.special_tokens_map = {
            "bos_token": self.bos_token if self.bos_token else "",
            "eos_token": self.eos_token if self.eos_token else "",
            "pad_token": self.pad_token if self.pad_token else "",
            "unk_token": self.unk_token if self.unk_token else ""
        }
        
        self.padding_side = config.get("padding_side", "right")
        
        
        