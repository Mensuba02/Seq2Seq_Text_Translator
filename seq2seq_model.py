

# Step 1: Install dependencies
!pip install -q gdown torch torchvision torchaudio

# Step 2: Download Aksharantar sample dataset
import gdown
url = "https://drive.google.com/uc?id=1tGIO4-IPNtxJ6RQMmykvAfY_B0AaLY5A"
output = "aksharantar_sampled.zip"
gdown.download(url, output, quiet=False)

# Step 3: Unzip dataset
!unzip -o aksharantar_sampled.zip

# Step 4: Load Hindi dataset (you can change language folder)
import pandas as pd
train_path = "/content/aksharantar_sampled/hin/hin_train.csv"
valid_path = "/content/aksharantar_sampled/hin/hin_valid.csv"
test_path  = "/content/aksharantar_sampled/hin/hin_test.csv"

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df  = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Columns:", train_df.columns)

# Step 5: Create character-level vocabulary manually
src_chars = set("".join(train_df['shastragaar'].values))
tgt_chars = set("".join(train_df['शस्त्रागार'].values))
special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
src_chars = special_tokens + sorted(list(src_chars))
tgt_chars = special_tokens + sorted(list(tgt_chars))

src_vocab = {char: idx for idx, char in enumerate(src_chars)}
tgt_vocab = {char: idx for idx, char in enumerate(tgt_chars)}
src_index2char = {idx: char for char, idx in src_vocab.items()}
tgt_index2char = {idx: char for char, idx in tgt_vocab.items()}

SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)

print("Source vocab size (romanized):", SRC_VOCAB_SIZE)
print("Target vocab size (Devanagari):", TGT_VOCAB_SIZE)

# Step 6: Prepare Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TransliterationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        src_word = list(self.df.iloc[idx]['shastragaar'])
        tgt_word = ["<sos>"] + list(self.df.iloc[idx]['शस्त्रागार']) + ["<eos>"]
        src_ids = torch.tensor([self.src_vocab.get(c, self.src_vocab["<unk>"]) for c in src_word], dtype=torch.long)
        tgt_ids = torch.tensor([self.tgt_vocab.get(c, self.tgt_vocab["<unk>"]) for c in tgt_word], dtype=torch.long)
        return src_ids, tgt_ids

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=src_vocab["<pad>"])
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab["<pad>"])
    return src_batch.to(device), tgt_batch.to(device)

train_dataset = TransliterationDataset(train_df, src_vocab, tgt_vocab)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

src_batch, tgt_batch = next(iter(train_loader))
print("Source batch shape:", src_batch.shape)
print("Target batch shape:", tgt_batch.shape)

# Step 7: Define Encoder, Decoder, Seq2Seq
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, rnn_type="gru"):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        else:
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers)
        self.rnn_type = rnn_type.lower()
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, rnn_type="gru"):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        else:
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.rnn_type = rnn_type.lower()
        
    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Step 8: Initialize model, loss, optimizer
INPUT_DIM = SRC_VOCAB_SIZE
OUTPUT_DIM = TGT_VOCAB_SIZE
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HID_DIM = 128
N_LAYERS = 1
RNN_TYPE = "gru"
LEARNING_RATE = 0.001
N_EPOCHS = 5  # small for demo, increase for better results

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, RNN_TYPE)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, RNN_TYPE)
model = Seq2Seq(enc, dec, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 9: Training loop
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch in train_loader:
        optimizer.zero_grad()
        output = model(src_batch, tgt_batch)
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        tgt = tgt_batch[1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")

# Step 10: Prediction function
def predict(model, word, src_vocab, tgt_vocab, tgt_index2char, max_len=30):
    model.eval()
    src_ids = torch.tensor([src_vocab.get(c, src_vocab["<unk>"]) for c in word], dtype=torch.long).unsqueeze(1).to(device)
    hidden = model.encoder(src_ids)
    input = torch.tensor([tgt_vocab["<sos>"]], dtype=torch.long).to(device)
    result = []
    for _ in range(max_len):
        output, hidden = model.decoder(input, hidden)
        top1 = output.argmax(1).item()
        if tgt_index2char[top1] == "<eos>":
            break
        result.append(tgt_index2char[top1])
        input = torch.tensor([top1], dtype=torch.long).to(device)
    return "".join(result)

# Step 11: Test some examples
test_words = ["ghar", "namaste", "dost"]
for w in test_words:
    print(f"Input word: {w} | Predicted Hindi: {predict(model, w, src_vocab, tgt_vocab, tgt_index2char)}")

# Model hyperparameters
m = 64         # embedding size
k = 128        # hidden size
T = 8          # average sequence length
V_src = 30     # source vocab size
V_tgt = 68     # target vocab size

# ---- Total Parameters ----
encoder_emb_params = V_src * m
decoder_emb_params = V_tgt * m
gru_params = 3 * (k**2 + k*m + k)  # 1 layer
total_gru_params = gru_params * 2   # encoder + decoder
decoder_out_params = k * V_tgt + V_tgt

total_params = encoder_emb_params + decoder_emb_params + total_gru_params + decoder_out_params

# ---- Total Computations ----
encoder_computations = 3 * T * (k**2 + k*m + k)
decoder_computations = 3 * T * (k**2 + k*m + k) + T * (k*V_tgt)
total_computations = encoder_computations + decoder_computations

# ---- Print Results ----
print("Total Parameters =", total_params)
print("Total Computations per sequence =", total_computations)

