import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

!gdown --id 1tGIO4-IPNtxJ6RQMmykvAfY_B0AaLY5A -O /content/aksharantar_sampled.zip

!unzip -q /content/aksharantar_sampled.zip -d /content/

!ls /content/

!ls /content/aksharantar_sampled

train_path = "/content/aksharantar_sampled/hin/hin_train.csv"
valid_path = "/content/aksharantar_sampled/hin/hin_valid.csv"
test_path = "/content/aksharantar_sampled/hin/hin_test.csv"

import pandas as pd
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df = pd.read_csv(test_path)
print("Train shape:", train_df.shape)
print("Columns:", train_df.columns)

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
src_ids = torch.tensor([self.src_vocab.get(c, self.src_vocab["<unk>"]) for c in src_word], 
dtype=torch.long)
tgt_ids = torch.tensor([self.tgt_vocab.get(c, self.tgt_vocab["<unk>"]) for c in tgt_word], 
dtype=torch.long)
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

import torch.optim as optim
INPUT_DIM = SRC_VOCAB_SIZE
OUTPUT_DIM = TGT_VOCAB_SIZE
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HID_DIM = 128
N_LAYERS = 1
RNN_TYPE = "gru"
LEARNING_RATE = 0.001
N_EPOCHS = 10
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, RNN_TYPE)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, RNN_TYPE)
model = Seq2Seq(enc, dec, device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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

def predict(model, word, src_vocab, tgt_vocab, tgt_index2char, max_len=30):
model.eval()
src_ids = torch.tensor([src_vocab.get(c, src_vocab["<unk>"]) for c in word], 
dtype=torch.long).unsqueeze(1).to(device)
hidden = model.encoder(src_ids)
input = torch.tensor([tgt_vocab["<sos>"]], dtype=torch.long).to(device)
result = [ ]
for _ in range(max_len):
output, hidden = model.decoder(input, hidden)
top1 = output.argmax(1).item()
if tgt_index2char[top1] == "<eos>":
break
result.append(tgt_index2char[top1])
input = torch.tensor([top1], dtype=torch.long).to(device)
return "".join(result)
word = " ghar" # Example word 
print("Input word:", word)
print("Predicted Hindi:", predict(model, word, src_vocab, tgt_vocab, tgt_index2char))

#To find the parameters

V_src = 30   
V_tgt = 68   
m = 64       
k = 128      

encoder_embed_params = V_src * m
decoder_embed_params = V_tgt * m

encoder_gru_params = 3 * (k**2 + k*m + k)
decoder_gru_params = 3 * (k**2 + k*m + k)

linear_params = k * V_tgt + V_tgt

total_params = encoder_embed_params + decoder_embed_params + encoder_gru_params + decoder_gru_params + linear_params

print("Encoder embedding params:", encoder_embed_params)
print("Decoder embedding params:", decoder_embed_params)
print("Encoder GRU params:", encoder_gru_params)
print("Decoder GRU params:", decoder_gru_params)
print("Linear layer params:", linear_params)
print("Total network parameters:", total_params)

#To find the computations

T = 8  
m = 64 
k = 128 
V = 68  

gru_computations = 2 * T * 3 * (k**2 + k*m + k)

linear_computations = T * (k * V + V)

total_computations = gru_computations + linear_computations

print("GRU computations:", gru_computations)
print("Linear layer computations:", linear_computations)
print("Total computations:", total_computations)

