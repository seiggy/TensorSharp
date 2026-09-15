"""Regenerate tokenizer-only oracles; no model weights or GPUs are needed.

Requires transformers, tokenizers, sentencepiece and huggingface_hub.
Use the slow SentencePiece tokenizer for XLM-R, matching the normalizer exported
into GGUF. Fast-tokenizer differences are retained separately, not hidden.
"""
import argparse
import importlib.metadata
import subprocess
import json
from pathlib import Path

from huggingface_hub import model_info
from transformers import AutoTokenizer

TEXTS = [
    "Hello world!",
    "public async Task<int> FindAsync(string query) => await Search(query);",
    "  spaces\tand\nnewlines  ",
    "Café naïve coöperate résumé",
    "Cafe\u0301 nai\u0308ve",
    "全角ＡＢＣ ① ﬁ ﬃ ㍿",
    "你好，世界！日本語と한국어",
    "مرحبا بالعالم שלום עולם",
    "éx\u0000\u0001\u200b\u00a0test",
    "🐻‍❄️ emoji 🙂𠮷",
    "", "[CLS] Hello [SEP]", "<s> Hello </s>",
    "query: where are the cache eviction methods?",
    "before <mask> after", "[MASK] AFTER [UNK] before",
    "<unk> <pad> literal </s><s>",
    "İstanbul Straße ẞ ΑΣ ςσ ΣΟΣ",
    "الْعَرَبِيَّةُ שָׁלוֹם हिन्दी বাংলা ไทย",
    "\u202f\u2009\u200a spaced\u3000text\ufeff",
    "x == null ? Array.Empty<byte>() : Encoding.UTF8.GetBytes(x);",
    'SELECT path, cosine_similarity(vector, ?) FROM embeddings ORDER BY 2 DESC;',
    'fn main() { println!("Hello, 世界! 🦀"); }',
    "foo_bar fooBar HELLO-WORLD std::vector<int> λx.x",
    "0 12 345 6.78 1e-10 -0.0 ٠١٢٣ １２３４",
    "é e\u0301 K K Å Å \ufb00 \ufb01 \ufb02",
    "🐻🐻🦀💻🧑🏽‍💻 a\u200db z\u200cq",
    "<s><s></s></s> [CLS][SEP][MASK]",
    " \t\r\n ",
    "a\r\nb\nc\td\ve\ff",
]
MODELS = {
    "snowflake": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
parser = argparse.ArgumentParser()
parser.add_argument("--llama-tokenize", required=True)
parser.add_argument("--model-dir", required=True)
args = parser.parse_args()
files = {"snowflake": "snowflake-arctic-embed-l-v2.0-q8_0.gguf",
         "minilm": "all-MiniLM-L6-v2-Q8_0.gguf"}
result = {}
for name, repo in MODELS.items():
    revision = model_info(repo).sha
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=False)
    fast = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
    cases = []
    for text in TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=True)
        case = {"text": text, "ids": ids}
        fast_ids = fast.encode(text, add_special_tokens=True)
        if fast_ids != ids:
            case["fast_tokenizer_ids"] = fast_ids
        llama = subprocess.run(
            [args.llama_tokenize, "-m", str(Path(args.model_dir) / files[name]), "--stdin", "--ids"],
            input=text, text=True, capture_output=True, check=True)
        gguf_ids = json.loads(llama.stdout)
        if gguf_ids != ids:
            case["llama_ids"] = gguf_ids
            if name == "snowflake":
                case["gguf_ids"] = gguf_ids
                case["difference"] = "This Snowflake GGUF omits the literal mask token from its vocabulary."
            elif "\v" in text or "\f" in text:
                case["gguf_ids"] = gguf_ids
                case["difference"] = "GGUF/llama WordPiece treats vertical-tab and form-feed as whitespace; HF drops them."
            else:
                case["difference"] = "llama.cpp simplified NFD drops decomposed Hangul components or spacing vowel marks; TensorSharp follows HuggingFace Unicode NFD."
        cases.append(case)
    result[name] = {
        "repo": repo, "revision": revision,
        "tokenizer_class": type(tokenizer).__name__, "use_fast": False,
        "packages": {key: importlib.metadata.version(key)
                     for key in ("transformers", "sentencepiece", "tokenizers")},
        "cases": cases,
    }
Path(__file__).with_name("huggingface-tokenization.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
