import sys
import logging
import json
import regex as re
from io import open
import itertools
from datetime import datetime
import os

import torch


PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'gpt2': 1024,
}


logger = logging.getLogger(__name__)


class Tokenization(object):
    """
    Tokenization object to hold tokenization, (processed text),and original
    text. Can hold tokenization as Ids or tokens.

    It also holds command tokens (pad, unk, etc.) for the tokenization.
    This allows functions to pad/operate on tokenization without having
    access to the full tokenizer, just the tokenization.

    Several standard array operations are implemented (insert, append, extend).
    """

    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, as_ids=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.as_ids = as_ids
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.as_ids:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def __str__(self):
        return f"Tokenization = {self.tokenization}, Text = {self.text}"

    def insert(self, idx, other):
        if isinstance(other, CommandToken):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text = other.token + self.text
                self.original_text = other.token + self.original_text
            elif idx == len(self.tokenization) - 1:
                self.text += other.token
                self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, CommandToken):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, CommandToken):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, list) and isinstance(other[0], CommandToken):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self


#@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class GPT2Tokenizer(object):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level BPE
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path == "roberta":
            vocab_file = BPE_VOCAB
            merges_file = BPE_MERGES
        # redirect to the cache, if necessary
        # try:
        #     resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        #     resolved_merges_file = cached_path(merges_file, cache_dir=cache_dir)
        # except EnvironmentError:
        #     logger.error(
        #         "Model name '{}' was not found in model name list ({}). "
        #         "We assumed '{}' was a path or url but couldn't find files {} and {} "
        #         "at this path or url.".format(
        #             pretrained_model_name_or_path,
        #             ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
        #             pretrained_model_name_or_path,
        #             vocab_file, merges_file))
        #     return None
        # if resolved_vocab_file == vocab_file and resolved_merges_file == merges_file:
        #     logger.info("loading vocabulary file {}".format(vocab_file))
        #     logger.info("loading merges file {}".format(merges_file))
        # else:
        #     logger.info("loading vocabulary file {} from cache at {}".format(
        #         vocab_file, resolved_vocab_file))
        #     logger.info("loading merges file {} from cache at {}".format(
        #         merges_file, resolved_merges_file))
        logger.info("loading vocabulary file {}".format(vocab_file))
        logger.info("loading merges file {}".format(merges_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(vocab_file, merges_file, *inputs, **kwargs)
        return tokenizer

    def __init__(self, vocab_file, merges_file, errors='replace', special_tokens=None, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    @property
    def tokens(self):
        return self.decoder

    @property
    def vocab(self):
        return self.encoder

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v:k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        merge_file = os.path.join(vocab_path, MERGES_NAME)
        special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        index = len(self.encoder)
        with open(special_tokens_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving special tokens vocabulary to {}: BPE indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(special_tokens_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        return vocab_file, merge_file, special_tokens_file


#def prep_command_tokens(tokenlist, token_format=token_format):
#    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __repr__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer, command_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)
        #print(command_tokens)
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self.command_tokens}
        self.command_token_map = {tok.token: tok for tok in self.command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self.command_tokens}

        # parse tokens and vocabs from tokenizer
        max_token_id = max(len(self.text_tokenizer.tokens) - 1, max(self.command_id_map.keys()))
        self._tokens = [self.text_tokenizer.tokens[i] if i < len(self.text_tokenizer.tokens) else f'[UNUSED{i}]' for i
                        in range(max_token_id + 1)]
        for idx, token in self.command_id_map.items():
            self._tokens[idx] = token.token
        self._vocab = {t.token: Id for Id, t in self.command_id_map.items()}
        self._vocab.update(self.text_tokenizer.vocab)

        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self.command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = len(self.tokens)

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t: Id for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self.spaces_between_special_tokens = True

    @property
    def command_tokens(self):
        return self._command_tokens

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._encode(token) if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        raise NotImplementedError

    def _decode(self, ids):
        raise NotImplementedError

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        return out_string

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.EncodeAsIds(text, process_fn=process_fn)
        tokenization.tokenization = [self.IdToToken(idx) for idx in tokenization.tokenization]
        return tokenization

    def IdToToken(self, idx):
        """convert Id to token accounting for command tokens"""
        if isinstance(idx, CommandToken):
            return idx.token
        return self.tokens[idx]

    def TokenToId(self, token):
        """convert token to Id accounting for command tokens"""
        if isinstance(token, CommandToken):
            return token.Id
        return self.vocab[token]

    def DecodeIds(self, ids):
        """
        convert Ids to tokens accounting for command tokens, tokens
        are joined and returned as a string.
        """
        rtn_strs = []
        current_str = []
        if isinstance(ids, Tokenization):
            ids = ids.tokenization
        for Id in ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self._decode(current_str))
                current_str = []
                rtn_strs.append(Id.token)
            elif Id in self.command_id_map:
                rtn_strs.append(self._decode(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id)
        if current_str:
            rtn_strs.append(self._decode(current_str))
        if self.spaces_between_special_tokens:
            output = ' '.join(rtn_strs)
        else:
            output = "".join(rtn_strs)
        output = self.clean_up_tokenization(output)
        return output

    def DecodeTokens(self, tokens):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        Ids = [self.TokenToId(token) for token in tokens]
        return self.DecodeIds(Ids)



class GPT2BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path, cache_dir=None, add_block_symbols=False, add_task_mask=False,
                 add_decoder_mask=False, **kwargs):
        text_tokenizer = GPT2Tokenizer.from_pretrained(model_type_or_path,
                                                       cache_dir=cache_dir)

        # disable max len warnings by increasing max len
        text_tokenizer.max_len = int(1e12)
        num_tokens = len(text_tokenizer.encoder)
        if model_type_or_path.startswith('roberta'):
            command_tokens = [
                CommandToken('pad', '<|endoftext|>', text_tokenizer.encoder['</s>']),
                CommandToken('eos', '<|endoftext|>', text_tokenizer.encoder['</s>']),
                CommandToken('sep', '[SEP]', text_tokenizer.encoder['<pad>']),
                CommandToken('ENC', '[CLS]', text_tokenizer.encoder['<s>']),
                CommandToken('MASK', '[MASK]', text_tokenizer.encoder['<mask>'], lstrip=True),
                CommandToken('unk', '[UNK]', text_tokenizer.encoder['<unk>'])
            ]
            if add_block_symbols:
                command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', num_tokens),
                    CommandToken('eop', '<|endofpiece|>', num_tokens + 1)
                ])
                num_tokens += 2
        else:
            command_tokens = [
                CommandToken('pad', '<|endoftext|>', text_tokenizer.encoder['<|endoftext|>']),
                CommandToken('eos', '<|endoftext|>', text_tokenizer.encoder['<|endoftext|>'])
            ]
            if add_block_symbols:
                command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', num_tokens),
                    CommandToken('eop', '<|endofpiece|>', num_tokens + 1),
                    CommandToken('ENC', '[CLS]', num_tokens + 2),
                    CommandToken('MASK', '[MASK]', num_tokens + 3, lstrip=True),
                    CommandToken('sep', '[SEP]', num_tokens + 4),
                    CommandToken('unk', '[UNK]', num_tokens + 5)
                ])
                num_tokens += 6
        if add_block_symbols:
            if add_task_mask:
                command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', num_tokens, lstrip=True),
                    CommandToken('sMASK', '[sMASK]', num_tokens + 1, lstrip=True)
                ])
                num_tokens += 2
            if add_decoder_mask:
                command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', num_tokens)
                ])
                num_tokens += 1
        super().__init__(text_tokenizer, command_tokens=command_tokens)

    def _encode(self, text):
        return self.text_tokenizer.encode(text)

    def _decode(self, ids):
        return self.text_tokenizer.decode(ids)


def get_tokenizer(cfg):
    '''
        If you're using outer_tokenizer, call `get_tokenizer(args, outer_tokenizer)`
        before `training_main`.
    '''
    # TODO (mchoi): tokenizer_model_type
    tokenizer_type = cfg.swiss_tokenization.tokenizer_type
    tokenizer_model_type = cfg.swiss_tokenization.tokenizer_model_type

    if torch.distributed.get_rank() == 0:
        logger.info(f"Building {cfg.swiss_tokenization.tokenizer_model_type} "
                    f"tokenizer: {tokenizer_type}")

    if tokenizer_type is None:
        raise Exception("No valid tokenizer type")

    # load the tokenizer according to tokenizer_type
    if tokenizer_type.startswith('cogview'):
        raise NotImplementedError
        get_tokenizer.tokenizer = UnifiedTokenizer(
            cfg.swiss_tokenization.img_tokenizer_path,
            txt_tokenizer_type='cogview',
            device=torch.cuda.current_device()
        )

    elif tokenizer_type.startswith('glm'):
        # TODO: args.task_mask is not an argument, only present in GLM
        kwargs = {
            "add_block_symbols": True,
            "add_task_mask": cfg.model.task_mask,
            "add_decoder_mask": cfg.model.block_mask_prob > 0.0
        }
        if tokenizer_type == "glm_GPT2BPETokenizer":
            get_tokenizer.tokenizer = GPT2BPETokenizer(tokenizer_model_type, **kwargs)

        elif tokenizer_type == "glm_ChineseSPTokenizer":
            raise NotImplementedError
            get_tokenizer.tokenizer = ChineseSPTokenizer(tokenizer_type, **kwargs)

    elif tokenizer_type == 'icetk':
        raise NotImplementedError
        get_tokenizer.tokenizer = icetk

    elif tokenizer_type == 'icetk-glm-130B':
        raise NotImplementedError
        get_tokenizer.tokenizer = _IceTokenizer()
    # elif tokenizer_type.startswith('hf'):
    #     from .hf_tokenizer import HFT5Tokenizer
    #     if tokenizer_type == "hf_T5Tokenizer":
    #         get_tokenizer.tokenizer = HFT5Tokenizer(args.tokenizer_model_type, cache_dir=args.cache_dir)
    else:
        raise NotImplementedError

        logger.info('Try to load tokenizer from Huggingface transformers...')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        from transformers import AutoTokenizer
        try:
            get_tokenizer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        except OSError as e:
            logger.info(f'Cannot find {tokenizer_type} from Huggingface or SwissArmyTransformer. Creating a fake tokenizer...')
            assert cfg.model.vocab_size > 0
            get_tokenizer.tokenizer = FakeTokenizer(cfg.model.vocab_size)
            return get_tokenizer.tokenizer

    logger.info(f'Set tokenizer as a {tokenizer_type} tokenizer! Now you can get_tokenizer() everywhere.')
    return get_tokenizer.tokenizer


class FakeTokenizer(object):
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens

    def __len__(self):
        return self.num_tokens


def timed_name(prefix, suffix=None, path=None):
    return os.path.join(
        path, 
        f"{prefix}-{datetime.now().strftime('%m-%d-%H-%M-%S')}{suffix}"
    )


