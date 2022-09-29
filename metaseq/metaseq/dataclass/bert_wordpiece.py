from dataclasses import dataclass, field

from metaseq import file_utils
from metaseq.data.encoders import register_tokenizer
from metaseq.dataclass import MetaseqDataclass


@dataclass
class BertWordPieceConfig(MetaseqDataclass):
    wordpiece_vocab: str = field(
        default="???", metadata={"help": "Path to vocab file"})


@register_tokenizer("bert_wordpiece", dataclass=BertWordPieceConfig)
class BertWordPiece:
    def __init__(self, cfg):
        try:
            from tokenizers import WordPiece
        except ImportError:
            raise ImportError(
                "Pleast install huggingface/tokenizers with: " "pip install tokenizers"
            )

        wordpiece_vocab = file_utils.cached_path(cfg.wordpiece_vocab)

        self.wordpiece = WordPiece(wordpiece_vocab)

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.wordpiece.encode(x).ids))

    def decode(self, x: str) -> str:
        raise NotImplementedError

    def is_beginning_of_word(self, x: str) -> bool:
        raise NotImplementedError
