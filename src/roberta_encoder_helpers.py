import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel


def get_roberta_tokenizer(
    model_name: str = "roberta-base",
    special_tokens: list = ["[NUM]"]
) -> BertTokenizer:
    """
    Load a RoBERTa tokenizer and add custom special tokens.

    Args:
        special_tokens: List of additional special tokens to add.

    Returns:
        A BertTokenizer with the added special tokens.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    return tokenizer


class RobertaEncoder(nn.Module):
    """
    Wraps a pre-trained Chinese RoBERTa model and projects its hidden states
    to 512 dimensions for compatibility with the existing tree decoder.

    Forward signature:
        input_ids: Tensor(B, S)
        attention_mask: Tensor(B, S)
    Returns:
        encoder_outputs: Tensor(S, B, 512)
        root_representation: Tensor(B, 512)
    """
    def __init__(
        self,
        model_name: str = "roberta-base",
        target_hidden_size: int = 512,
        max_length: int = 128,
        device: str = "cpu",
        special_tokens: list = ["[NUM]"]
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length

        # Load RoBERTa and add special tokens
        self.tokenizer = get_roberta_tokenizer(model_name, special_tokens)
        self.roberta   = RobertaModel.from_pretrained(model_name)
        self.roberta.resize_token_embeddings(len(self.tokenizer))

        # Linear projection from RoBERTa hidden size to target_hidden_size
        self.project = nn.Linear(
            self.roberta.config.hidden_size,
            target_hidden_size
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor
    ):
        # Encode with RoBERTa
        outputs = self.roberta(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        # last_hidden_state: (B, S, H_roberta)
        last_hidden = outputs.last_hidden_state
        # Project to (B, S, 512)
        proj_hidden = self.project(last_hidden)
        # Transpose to (S, B, 512)
        encoder_outputs = proj_hidden.transpose(0, 1)

        # Pooler output ([CLS]): (B, H_roberta) -> project to (B, 512)
        pooler = outputs.pooler_output
        root_representation = self.project(pooler)

        return encoder_outputs, root_representation


def prepare_roberta_batch(
    token_seqs: list,
    tokenizer: BertTokenizer,
    max_length: int = 128,
    device: str = "cpu"
):
    """
    Tokenize and batch raw token lists for RoBERTa.

    Args:
        token_seqs: List of token lists (e.g. [["我","有","[NUM]"], ...]).
        tokenizer: RoBERTa tokenizer with added special tokens.
        max_length: Maximum sequence length for padding/truncation.
        device: Device string for tensors.

    Returns:
        input_ids: Tensor of shape (B, S)
        attention_mask: Tensor of shape (B, S)
    """
    # Join tokens into whitespace-separated strings
    texts = [" ".join(seq) for seq in token_seqs]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask


class RobertaEncoderAdapter(nn.Module):
    """
    Adapter to seamlessly replace the BiLSTM encoder with RoBERTa.

    forward(input_seqs: Tensor[S,B], input_lengths: list) -> (encoder_outputs, root)
    where:
      - input_seqs: Tensor of word indices (S, B)
      - input_lengths: List of actual lengths per batch element
    """
    def __init__(self, input_lang, roberta_encoder: RobertaEncoder):
        super().__init__()
        self.input_lang = input_lang  # Lang object mapping indices back to words
        self.roberta = roberta_encoder
        self.hidden_size = roberta_encoder.project.out_features

    def forward(
        self,
        input_seqs: torch.LongTensor,
        input_lengths: list,
        hidden_init=None
    ):
        # Reconstruct token lists from indices
        batch_size, _ = len(input_lengths), input_seqs.size(1)
        sequences = input_seqs.transpose(0, 1)  # (B, S)
        token_seqs = []
        for i, L in enumerate(input_lengths):
            ids = sequences[i, :L].tolist()
            tokens = [self.input_lang.index2word[idx] for idx in ids]
            token_seqs.append(tokens)

        # Tokenize and encode
        input_ids, attention_mask = prepare_roberta_batch(
            token_seqs,
            self.roberta.tokenizer,
            max_length=self.roberta.max_length,
            device=self.roberta.device
        )
        return self.roberta(input_ids, attention_mask)


def transfer_num_roberta(data):
    """
    Wrapper to mask numbers and generate (input_seq, out_seq, nums, num_pos) pairs.

    Args:
        data: Either a path to raw JSON or a pre-loaded list of examples.
    Returns:
        List of tuples (input_seq, out_seq, nums, num_pos).
    """
    from pre_data import load_raw_data, transfer_num
    raw = load_raw_data(data) if isinstance(data, str) else data
    return transfer_num(raw)


def setup_roberta_training(params, input_lang):
    """
    Initialize RoBERTa encoder & adapter, decoder, and optimizer.

    Args:
        params: Namespace with model hyperparameters (model_name, hidden_size,
                max_input_len, device, etc.).
        input_lang: Lang object for input vocabulary.
    Returns:
        encoder, decoder, optimizer
    """
    from train_and_evaluate import define_decoder, define_optimizer

    # Instantiate and wrap RoBERTa
    roberta = RobertaEncoder(
        model_name=params.model_name,
        target_hidden_size=params.hidden_size,
        max_length=params.max_input_len,
        device=params.device
    )
    encoder = RobertaEncoderAdapter(input_lang, roberta).to(params.device)

    # Decoder and optimizer remain unchanged
    decoder = define_decoder(params)
    optimizer = define_optimizer(params, encoder, decoder)

    return encoder, decoder, optimizer







# import torch
# import torch.nn as nn
# from transformers import BertTokenizer, BertModel


# def get_roberta_tokenizer(
#     model_name: str = "hfl/chinese-roberta-wwm-ext",
#     special_tokens: list = ["[NUM]"]
# ) -> BertTokenizer:
#     """
#     Load a Chinese RoBERTa tokenizer and add custom special tokens.
#     """
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
#     return tokenizer


# class RobertaEncoder(nn.Module):
#     """
#     Wraps a pre-trained Chinese RoBERTa model and projects its hidden states
#     to a fixed size for compatibility with the existing tree decoder.
#     """

#     def __init__(
#         self,
#         model_name: str = "hfl/chinese-roberta-wwm-ext",
#         target_hidden_size: int = 512,
#         max_length: int = 128,
#         device: str = "cpu",
#         special_tokens: list = ["[NUM]"]
#     ):
#         super().__init__()
#         self.device = device
#         # Load pre-trained RoBERTa
#         self.roberta = BertModel.from_pretrained(model_name)
#         # Prepare tokenizer with special tokens
#         self.tokenizer = get_roberta_tokenizer(model_name, special_tokens)
#         # Resize embeddings if new tokens were added
#         self.roberta.resize_token_embeddings(len(self.tokenizer))
#         # Linear projection: RoBERTa hidden_size -> target_hidden_size
#         self.project = nn.Linear(
#             self.roberta.config.hidden_size,
#             target_hidden_size
#         )
#         self.max_length = max_length

#     def forward(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: torch.LongTensor
#     ):
#         # RB encoding
#         outputs = self.roberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask.to(self.device)
#         )
#         last_hidden = outputs.last_hidden_state  # (B, S, H)
#         proj_hidden = self.project(last_hidden)  # (B, S, target_hidden_size)
#         encoder_outputs = proj_hidden.transpose(0, 1)  # (S, B, H)
#         pooler = outputs.pooler_output  # (B, H)
#         root = self.project(pooler)      # (B, H)
#         return encoder_outputs, root


# def prepare_roberta_batch(
#     token_seqs: list,
#     tokenizer: BertTokenizer,
#     max_length: int = 128,
#     device: str = "cpu"
# ):
#     """
#     Convert a list of token lists into RoBERTa input tensors.

#     Args:
#         token_seqs: List of token lists (e.g. [["我","有","[NUM]"], ...]).
#         tokenizer: RoBERTa tokenizer with added special tokens.
#     Returns:
#         input_ids: Tensor (B, S)
#         attention_mask: Tensor (B, S)
#     """
#     enc = tokenizer(
#         [" ".join(seq) for seq in token_seqs],
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )
#     input_ids = enc["input_ids"].to(device)
#     attention_mask = enc["attention_mask"].to(device)
#     return input_ids, attention_mask


# class RobertaEncoderAdapter(nn.Module):
#     """
#     Adapter to mimic BaselineEncoderLSTM interface using RoBERTa.

#     forward(input_seqs, input_lengths) -> (encoder_outputs, root)
#     where:
#       - input_seqs: S x B LongTensor of word-index ids
#       - input_lengths: list of actual lengths per batch item
#     """
#     def __init__(
#         self,
#         input_lang,
#         roberta_encoder: RobertaEncoder
#     ):
#         super().__init__()
#         self.input_lang = input_lang  # Lang object with index2word
#         self.roberta = roberta_encoder

#     def forward(self, input_seqs: torch.LongTensor, input_lengths: list, hidden_init=None):
#         # Reconstruct token lists
#         # input_seqs: (S, B)
#         sequences = input_seqs.transpose(0,1)  # (B, S)
#         token_seqs = []
#         for i, length in enumerate(input_lengths):
#             ids = sequences[i, :length].tolist()
#             words = [self.input_lang.index2word[idx] for idx in ids]
#             token_seqs.append(words)
#         # Prepare RoBERTa inputs
#         input_ids, attention_mask = prepare_roberta_batch(
#             token_seqs,
#             self.roberta.tokenizer,
#             max_length=self.roberta.max_length,
#             device=self.roberta.device
#         )
#         # Encode
#         return self.roberta(input_ids, attention_mask)

# import torch.nn as nn
# from transformers import BertTokenizer, BertModel


# def get_roberta_tokenizer(
#     model_name: str = "hfl/chinese-roberta-wwm-ext",
#     special_tokens: list = ["[NUM]"]
# ) -> BertTokenizer:
#     """
#     Load a Chinese RoBERTa tokenizer and add custom special tokens.

#     Args:
#         model_name: HuggingFace model identifier for the Chinese RoBERTa.
#         special_tokens: List of tokens to add (e.g., "[NUM]").

#     Returns:
#         A BertTokenizer with added special tokens.
#     """
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
#     return tokenizer


# class RobertaEncoder(nn.Module):
#     """
#     Wraps a pre-trained Chinese RoBERTa model and projects its hidden states
#     to a fixed size for compatibility with the existing tree decoder.

#     The forward pass returns:
#       - encoder_outputs: (S, B, hidden_size)
#       - root_representation: (B, hidden_size)
#     """

#     def __init__(
#         self,
#         model_name: str = "hfl/chinese-roberta-wwm-ext",
#         target_hidden_size: int = 512
#     ):
#         super().__init__()
#         # Load pre-trained RoBERTa
#         self.roberta = BertModel.from_pretrained(model_name)
#         # Prepare tokenizer with special tokens
#         self.tokenizer = get_roberta_tokenizer(model_name)
#         # Resize embeddings if new tokens were added
#         self.roberta.resize_token_embeddings(len(self.tokenizer))
#         # Linear projection: RoBERTa hidden_size -> target_hidden_size
#         self.project = nn.Linear(
#             self.roberta.config.hidden_size,
#             target_hidden_size
#         )

#     def forward(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: torch.LongTensor
#     ):
#         # RoBERTa encoding
#         outputs = self.roberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         # last_hidden_state: (B, S, H_roberta)
#         last_hidden = outputs.last_hidden_state
#         # Project to decoder hidden size
#         proj_hidden = self.project(last_hidden)      # (B, S, target_hidden_size)
#         # Transpose to (S, B, target_hidden_size)
#         encoder_outputs = proj_hidden.transpose(0, 1)
#         # Pooler output for [CLS] token: (B, H_roberta)
#         pooler = outputs.pooler_output
#         # Project to root representation: (B, target_hidden_size)
#         root = self.project(pooler)
#         return encoder_outputs, root


# def prepare_roberta_batch(
#     pairs: list,
#     tokenizer: BertTokenizer,
#     max_length: int = 128,
#     device: str = "cpu"
# ):
#     """
#     Convert tokenized examples into RoBERTa input tensors.

#     Args:
#         pairs: List of tuples where pairs[i][0] is the list of tokens
#                (with "[NUM]" placeholders) for example i.
#         tokenizer: RoBERTa tokenizer with added special tokens.
#         max_length: Maximum token sequence length (pad/truncate).
#         device: Device to place tensors on.

#     Returns:
#         input_ids: Tensor of shape (B, S)
#         attention_mask: Tensor of shape (B, S)
#         original_lengths: List of actual lengths per example
#     """
#     # Reconstruct text by joining token list with spaces
#     texts = [" ".join(seq[0]) for seq in pairs]
#     enc = tokenizer(
#         texts,
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )
#     input_ids = enc["input_ids"].to(device)
#     attention_mask = enc["attention_mask"].to(device)
#     # Real lengths before padding
#     original_lengths = attention_mask.sum(dim=1).tolist()
#     return input_ids, attention_mask, original_lengths
