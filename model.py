import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm
import requests
import os
import tempfile

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        output = self.combine_heads(output)
        output = self.W_o(output)

        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self attention with causal mask
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross attention with encoder output
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed forward
        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_padding_mask & tgt_sub_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc_out(decoder_output)

        return output

class UrduChatBot:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_response(self, input_text, max_length=50, temperature=0.8):
        # Tokenize input
        input_tokens = self.tokenizer.encode(input_text)
        src = torch.tensor([input_tokens]).to(self.device)
        src_mask = self.model.make_src_mask(src)

        # Start with SOS token
        tgt = torch.tensor([[self.tokenizer.bos_id()]]).to(self.device)

        for _ in range(max_length - 1):
            tgt_mask = self.model.make_tgt_mask(tgt)

            with torch.no_grad():
                encoder_output = self.model.encoder(src, src_mask)
                decoder_output = self.model.decoder(tgt, encoder_output, src_mask, tgt_mask)
                output = self.model.fc_out(decoder_output)

                # Get last token predictions
                next_token_logits = output[:, -1, :] / temperature
                probabilities = F.softmax(next_token_logits, dim=-1)

                # Sample from distribution
                next_token = torch.multinomial(probabilities, 1)

            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if EOS token generated
            if next_token.item() == self.tokenizer.eos_id():
                break

        # Decode response
        response_tokens = tgt[0].tolist()
        response = self.tokenizer.decode(response_tokens)

        # Remove SOS and EOS tokens from response
        response = response.replace('<s>', '').replace('</s>', '').strip()

        return response

def download_file(url, local_filename):
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_filename
    except Exception as e:
        raise Exception(f"Error downloading {url}: {str(e)}")

def load_model_and_tokenizer():
    """Load model and tokenizer from GitHub URLs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Your actual GitHub URLs
    VOCAB_URL = "https://github.com/Mustehsan-Nisar-Rao/Urdu-Chat-Bot/raw/main/urdu_span_spm.model"
    WEIGHTS_URL = "https://github.com/Mustehsan-Nisar-Rao/Urdu-Chat-Bot/releases/download/v.1/best_finetuned_urdu_chatbot.pth"
    
    # Create temporary directory for files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download vocab file
        print("Downloading tokenizer model...")
        vocab_path = os.path.join(temp_dir, "urdu_span_spm.model")
        download_file(VOCAB_URL, vocab_path)
        
        # Download weights file
        print("Downloading model weights...")
        weights_path = os.path.join(temp_dir, "model_weights.pth")
        download_file(WEIGHTS_URL, weights_path)
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(vocab_path)
        vocab_size = tokenizer.get_piece_size()
        print(f"Vocabulary size: {vocab_size}")
        
        # Model configuration based on your training
        config = {
            'd_model': 512,
            'num_layers': 4,
            'num_heads': 4,
            'd_ff': 1024,
            'dropout': 0.1,
            'max_len': 100
        }
        
        # Create model
        print("Creating model architecture...")
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_len=config['max_len']
        ).to(device)
        
        # Load weights
        print("Loading model weights...")
        checkpoint = torch.load(weights_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
        
        # Create chatbot instance
        chatbot = UrduChatBot(model, tokenizer, device)
        
        return chatbot

# For testing the model directly
if __name__ == "__main__":
    chatbot = load_model_and_tokenizer()
    
    # Test the chatbot
    test_inputs = [
        "ہیلو کیا حال ہے",
        "تمہارا نام کیا ہے",
        "کیا تم میری مدد کر سکتے ہو"
    ]
    
    for test_input in test_inputs:
        response = chatbot.generate_response(test_input)
        print(f"Input: {test_input}")
        print(f"Response: {response}")
        print("-" * 50)
