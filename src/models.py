# coding: utf-8

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        # hidden (query) is (B, H_query) e.g. (batch_size, current_node_hidden_size)
        # encoder_outputs (sequence) is (S, B, H_encoder) e.g. (seq_len, batch_size, encoder_hidden_size)
        
        seq_len = encoder_outputs.size(0) # S
        batch_size_from_hidden = hidden.size(0) # B from query
        batch_size_from_encoder = encoder_outputs.size(1) # B from sequence

        # Ensure batch sizes match (should always be true if called correctly)
        assert batch_size_from_hidden == batch_size_from_encoder, "Batch sizes of query and sequence mismatch in TreeAttn"

        # Expand hidden (query) to be (S, B, H_query) for concatenation with encoder_outputs
        # hidden starts as (B, H_query)
        query_expanded = hidden.unsqueeze(0).repeat(seq_len, 1, 1) # (1, B, H_query) -> (S, B, H_query)
        
        # Concatenate expanded query and encoder_outputs along feature dimension (dim 2)
        # query_expanded: (S, B, H_query)
        # encoder_outputs: (S, B, H_encoder)
        # Result energy_in_concat: (S, B, H_query + H_encoder)
        energy_in_concat = torch.cat((query_expanded, encoder_outputs), 2)
        
        # Reshape for the linear layer self.attn, which expects (N, H_query + H_encoder)
        # self.hidden_size in __init__ is H_query, self.input_size in __init__ is H_encoder.
        # So, the input dimension for self.attn is self.hidden_size + self.input_size.
        energy_in_reshaped = energy_in_concat.view(-1, self.hidden_size + self.input_size)

        score_feature = torch.tanh(self.attn(energy_in_reshaped)) # Output: (S*B, self.hidden_size or attn_internal_dim)
                                                              # Current self.attn outputs self.hidden_size (H_query)
        attn_energies = self.score(score_feature)  # self.score expects H_query, outputs (S*B, 1)
        attn_energies = attn_energies.squeeze(1) # (S*B)
        attn_energies = attn_energies.view(seq_len, batch_size_from_hidden).transpose(0, 1)  # (S, B) -> (B, S)
        
        if seq_mask is not None:
            # ——— GLOBAL seq_mask FIX ———
            # Make sure the BoolTensor mask (B×L_mask) matches attn_energies (B×seq_len)
            B, seq_len = attn_energies.shape
            mask = seq_mask
            if mask.size(1) < seq_len:
                pad = mask.new_ones(B, seq_len - mask.size(1))
                mask = torch.cat([mask, pad], dim=1)
            elif mask.size(1) > seq_len:
                mask = mask[:, :seq_len]
            attn_energies = attn_energies.masked_fill_(mask, -1e12)

        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # (B, S)

        return attn_energies.unsqueeze(1) # (B, 1, S)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)
        self.is_leaf = nn.Linear(hidden_size * 2, 2)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, rpkhs_merged_encoder_outputs, num_pades, rpkhs_root_representation_Y, seq_mask, mask_nums):
        # rpkhs_root_representation_Y is (B, hidden_size)
        # node_stacks is a list (length B) of lists (stacks of TreeNode)
        # TreeNode.embedding is (1, hidden_size)
        # left_childs is a list (length B) of Tensors (1, hidden_size) or None
        
        batch_size = len(node_stacks) # Or rpkhs_root_representation_Y.size(0)

        current_embeddings_for_loop = []
        for idx in range(batch_size):
            st_nodes = node_stacks[idx]
            if not st_nodes: # Stack is empty for this batch item
                # Use the root representation for this item, ensure shape (hidden_size)
                current_emb = rpkhs_root_representation_Y[idx]
            else:
                # TreeNode.embedding is (1, hidden_size), squeeze to (hidden_size)
                current_emb = st_nodes[-1].embedding.squeeze(0)
            current_embeddings_for_loop.append(current_emb)

        current_node_temp = []
        for i in range(batch_size):
            l_batch_item = left_childs[i]           # (1, hidden_size) or None
            c_batch_item = current_embeddings_for_loop[i] # (hidden_size)

            c_dropped = self.dropout(c_batch_item) # (hidden_size)

            if l_batch_item is None:
                # Input to concat_l is c_dropped (hidden_size)
                g = torch.tanh(self.concat_l(c_dropped))
                t = torch.sigmoid(self.concat_lg(c_dropped))
                current_node_temp.append(g * t) # Appends a (hidden_size) tensor
            else:
                # l_batch_item is (1, hidden_size)
                # c_dropped is (hidden_size). Need to make it (1, hidden_size) for dim=1 cat
                c_batch_item_unsqueezed = c_dropped.unsqueeze(0) # Now (1, hidden_size)
                
                l_batch_item_dropped = self.dropout(l_batch_item) # l_batch_item is already (1, hidden_size)
                
                # Concat l_batch_item_dropped (1, H) and c_batch_item_unsqueezed (1, H) along dim 1
                concatenated_input = torch.cat((l_batch_item_dropped, c_batch_item_unsqueezed), 1) # Results in (1, 2*hidden_size)
                
                output_r = self.concat_r(concatenated_input) # (1, hidden_size)
                output_rg = self.concat_rg(concatenated_input) # (1, hidden_size)
                
                g = torch.tanh(output_r)
                t = torch.sigmoid(output_rg)
                
                current_node_temp.append((g * t).squeeze(0)) # Squeeze to (hidden_size) before appending

        current_node = torch.stack(current_node_temp, dim=0) # Results in (B, hidden_size)

        current_embeddings = self.dropout(current_node) # current_embeddings is (B, H)

        # Pass current_embeddings (B,H) directly as the query to TreeAttn
        current_attn = self.attn(current_embeddings, rpkhs_merged_encoder_outputs, seq_mask)
        current_context = current_attn.bmm(rpkhs_merged_encoder_outputs.transpose(0, 1))  # current_context shape: (B, 1, H_encoder)

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        # current_node has shape (B, hidden_size)
        # current_context has shape (B, 1, H_encoder)
        # To concatenate along dim 2, current_node needs to be (B, 1, hidden_size)
        current_node_unsqueezed = current_node.unsqueeze(1) # Shape: (B, 1, hidden_size)
        
        leaf_input = torch.cat((current_node_unsqueezed, current_context), 2) # Shape: (B, 1, hidden_size + H_encoder)
        leaf_input = leaf_input.squeeze(1) # Shape: (B, hidden_size + H_encoder)
        leaf_input = self.dropout(leaf_input)

        # self.is_leaf = nn.Linear(hidden_size + encoder_hidden_size, 2)
        # p_leaf = F.log_softmax(self.is_leaf(leaf_input), dim=1)               # (B,2)

        p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        return p_leaf, num_score, op, current_node, current_context, embedding_weight
        # return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, num_operators, num_constants, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_operators = num_operators
        self.num_constants = num_constants

        self.embedding_op = nn.Embedding(num_operators, embedding_size, padding_idx=0)
        self.embedding_con = nn.Embedding(num_constants, embedding_size, padding_idx=0)
        
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, current_node_label, current_context):
        # current_node_label is BATCH_SIZE (tensor of indices)
        # node_embedding is BATCH_SIZE x 1 x HIDDEN_SIZE (parent node embedding)
        # current_context is BATCH_SIZE x 1 x HIDDEN_SIZE
        
        op_vec_embedded = torch.zeros(current_node_label.size(0), self.embedding_size, device=current_node_label.device)
        
        op_mask = current_node_label < self.num_operators
        # Assumes current_node_label for constants are in range [num_operators, num_operators + num_constants - 1]
        con_mask = (current_node_label >= self.num_operators) & (current_node_label < self.num_operators + self.num_constants)

        if op_mask.any():
            op_indices_to_embed = current_node_label[op_mask]
            op_vec_embedded[op_mask] = self.embedding_op(op_indices_to_embed)
        
        if con_mask.any():
            con_indices_from_label = current_node_label[con_mask]
            # Adjust index for the constant embedding layer, which is 0-indexed
            con_indices_for_embedding = con_indices_from_label - self.num_operators
            op_vec_embedded[con_mask] = self.embedding_con(con_indices_for_embedding)
        
        # This op_vec_embedded is the equivalent of node_label_ in the original code
        # It's the embedding of the current operator/constant symbol.
        
        # Apply dropout to the symbol embedding
        processed_node_symbol_embedding = self.em_dropout(op_vec_embedded) 

        # Prepare other inputs (ensure they are squeezed and dropped out as in original)
        squeezed_node_embedding = node_embedding.squeeze(1) # Parent embedding
        squeezed_current_context = current_context.squeeze(1) # Context from attention
        
        dropped_node_embedding = self.em_dropout(squeezed_node_embedding)
        dropped_current_context = self.em_dropout(squeezed_current_context)

        # Concatenate features to generate child embeddings
        # Original was torch.cat((node_embedding, current_context, node_label), 1)
        # where node_label was the result of self.em_dropout(self.embeddings(node_label))
        # So, use processed_node_symbol_embedding here.
        concat_features = torch.cat((dropped_node_embedding, dropped_current_context, processed_node_symbol_embedding), 1)

        l_child = torch.tanh(self.generate_l(concat_features))
        l_child_g = torch.sigmoid(self.generate_lg(concat_features))
        r_child = torch.tanh(self.generate_r(concat_features))
        r_child_g = torch.sigmoid(self.generate_rg(concat_features))
        
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        
        # Return left child, right child, and the (non-dropped out) embedding of the current op/constant symbol itself
        return l_child, r_child, op_vec_embedded


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class BaselineEncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(BaselineEncoderLSTM, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        # hidden_size for LSTM is the size of each direction.
        # The summed output will also have this hidden_size.
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,
                            dropout=dropout if n_layers > 1 else 0, # LSTM dropout only applies if n_layers > 1
                            bidirectional=True, batch_first=False) # Input: S x B x E

    def forward(self, input_seqs, input_lengths, hidden_init=None):
        # input_seqs: S x B (Sequence Length, Batch Size)
        # input_lengths: B
        input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.int64).cpu() # Ensure input_lengths is a CPU tensor
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)

        # Pack sequence; enforce_sorted=False is generally safer unless data is pre-sorted.
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths_tensor, enforce_sorted=False)

        # LSTM initial hidden state (h_0, c_0) is optional.
        # h_0 shape: (num_layers * num_directions, batch, hidden_size)
        # c_0 shape: (num_layers * num_directions, batch, hidden_size)
        lstm_outputs, (final_hidden_state, final_cell_state) = self.lstm(packed, hidden_init)

        # Unpack sequence
        # lstm_outputs: S x B x (hidden_size * num_directions)
        lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_outputs)

        # Sum bidirectional LSTM outputs to get the final token-level representations
        # Output shape: S x B x hidden_size
        rpkhs_merged_encoder_outputs = lstm_outputs[:, :, :self.hidden_size] + \
                                       lstm_outputs[:, :, self.hidden_size:]

        # Create the root representation for the decoder (rpkhs_root_representation_Y)
        # Use the final hidden states of the last LSTM layer.
        # final_hidden_state shape: (num_layers * num_directions, batch, hidden_size)
        # Last layer, forward direction: final_hidden_state[-2, :, :]
        # Last layer, backward direction: final_hidden_state[-1, :, :]
        # Sum them to get a B x hidden_size vector.
        root_representation_forward = final_hidden_state[-2, :, :]
        root_representation_backward = final_hidden_state[-1, :, :]
        rpkhs_root_representation_Y = root_representation_forward + root_representation_backward

        return rpkhs_merged_encoder_outputs, rpkhs_root_representation_Y
