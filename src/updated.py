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



import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalReasoningEncoder(nn.Module):
    """
    Implements the 2-layer GRU + attention hierarchical reasoning:
      • Word-level: BiGRU over tokens in each sentence + word-attention → sentence vectors
      • Sent-level: BiGRU over sentence vectors + sent-attention → Yh (B×H)
    References: EMNLP21 (Yu et al.) Sec 3.4, Eqs (8–15) :contentReference[oaicite:0]{index=0}
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Word-level BiGRU (bidirectional → hidden_size)
        self.word_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True
        )
        self.w_attn = nn.Linear(hidden_size, hidden_size)  # Eq (8)
        self.w_context = nn.Parameter(torch.randn(hidden_size))  # uw

        # Sentence-level BiGRU
        self.sent_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True
        )
        self.s_attn = nn.Linear(hidden_size, hidden_size)  # Eq (13)
        self.s_context = nn.Parameter(torch.randn(hidden_size))  # us

    def forward(self, token_feats: torch.Tensor, sentence_spans: list):
        """
        Args:
          token_feats: Tensor of shape (S, B, H)
          sentence_spans: List of length B; each element is
              a list of (start, end) token‐indices for each sentence in that example
        Returns:
          Yh: Tensor(B, H) — the hierarchical representation
        """
        # (S, B, H) → (B, S, H)
        token_feats = token_feats.transpose(0,1)
        B, S, H = token_feats.shape

        # 1) Word-level: for each example, run BiGRU + attention on each sentence
        sent_vecs = []  # will collect: for each example, a Tensor (n_sent, H)
        for b in range(B):
            spans = sentence_spans[b]
            feats_b = token_feats[b]          # (S, H)
            vecs = []
            for st, ed in spans:
                seg = feats_b[st:ed]          # (Ti, H)
                if seg.size(0)==0:
                    # edge case: empty span
                    vecs.append(torch.zeros(H, device=seg.device))
                    continue
                # BiGRU: need batch dimension → (1, Ti, H)
                out, _ = self.word_gru(seg.unsqueeze(0))  # (1, Ti, H)
                out = out.squeeze(0)                       # (Ti, H)
                # word‐attention (Eqs 8–10)
                u = torch.tanh(self.w_attn(out))          # (Ti, H)
                scores = (u @ self.w_context)             # (Ti)
                α = F.softmax(scores, dim=0)               # (Ti)
                s = (α.unsqueeze(1) * out).sum(dim=0)      # (H)
                vecs.append(s)
            sent_vecs.append(torch.stack(vecs, dim=0))      # (n_sent, H)

        # pad to a uniform n_sent_max
        n_sent_max = max(v.size(0) for v in sent_vecs)
        padded = []
        masks  = []
        for v in sent_vecs:
            n = v.size(0)
            pad = torch.zeros(n_sent_max-n, H, device=v.device)
            padded.append(torch.cat([v, pad], dim=0))     # (n_sent_max, H)
            masks.append(torch.tensor([1]*n + [0]*(n_sent_max-n),
                                      device=v.device, dtype=torch.bool))
        sent_feats = torch.stack(padded, dim=0)           # (B, n_sent_max, H)
        sent_mask  = torch.stack(masks, dim=0)            # (B, n_sent_max)

        # 2) Sentence-level BiGRU
        out_sent, _ = self.sent_gru(sent_feats)           # (B, n_sent_max, H)

        # 3) Sent‐attention (Eqs 13–15)
        u = torch.tanh(self.s_attn(out_sent))             # (B, n_sent_max, H)
        # score each sentence
        scores = (u @ self.s_context)                     # (B, n_sent_max)
        # mask padding
        scores = scores.masked_fill(~sent_mask, -1e12)
        α = F.softmax(scores, dim=1)                      # (B, n_sent_max)
        Yh = (α.unsqueeze(2) * out_sent).sum(dim=1)       # (B, H)

        return Yh


class RPKHSFusion(nn.Module):
    """
    Implements Eq (16–18):
      Y = F([ w_p * Yp , w_h * Yh ]) with
      w_p = softmax( φ_p(Yp Wp) , φ_h(Yh Wh) )
    References: EMNLP21 Sec 3.4, Eqs (16–18) :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # projection to score space
        self.Wp = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        # small MLPs φ_p, φ_h
        self.ϕp = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size//2, 1))
        self.ϕh = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size//2, 1))
        # final fusion
        self.F  = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, Yp: torch.Tensor, Yh: torch.Tensor):
        # scalar scores
        sp = self.ϕp(torch.tanh(self.Wp(Yp)))    # (B,1)
        sh = self.ϕh(torch.tanh(self.Wh(Yh)))    # (B,1)
        w = torch.softmax(torch.cat([sp, sh], dim=1), dim=1)  # (B,2)
        wp, wh = w[:,0:1], w[:,1:2]             # each (B,1)
        # weighted combine & final FC
        Y = self.F(torch.cat([wp*Yp, wh*Yh], dim=1))  # (B,H)
        return Y




class RPKHSEncoderAdapter(RobertaEncoderAdapter):
    def __init__(self, input_lang, roberta_encoder, hier_encoder, fusion):
        super().__init__(input_lang, roberta_encoder)
        self.hier = hier_encoder
        self.fuse = fusion

    def forward(self, input_seqs, input_lengths, sentence_spans):
        # 1) get token‐level features & pretrained root
        enc_out, root_p = super().forward(input_seqs, input_lengths)
        # 2) hierarchical reasoning
        Yh = self.hier(enc_out, sentence_spans)
        # 3) fuse
        Y  = self.fuse(root_p, Yh)
        return enc_out, Y
