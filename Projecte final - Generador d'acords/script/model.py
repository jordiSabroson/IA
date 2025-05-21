import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('../data/chordomicon_clean.csv')

# Hiperparàmetres
embedding_dim = 64
hidden_dim = 128
num_layers = 2
learning_rate = 1e-5
num_epochs = 10
batch_size = 16
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
# print(f"Using device: {device}")

# --- Definició del dataset ---
class ChordDataset(Dataset):
    def __init__(self, data):
        self.data = self._preprocess(data)
        self.chord_vocab = self._build_vocab(self.data['chords_tokenized'])
        self.section_vocab = sorted(list(set(self.data['sections'])))
        self.genre_vocab = sorted(list(set(self.data['genres']))) 
        self.chord_to_index = {chord: idx for idx, chord in enumerate(self.chord_vocab)}
        self.section_to_index = {section: idx for idx, section in enumerate(self.section_vocab)}
        self.genre_to_index = {genre: idx for idx, genre in enumerate(self.genre_vocab)}
        self.index_to_chord = {idx: chord for chord, idx in self.chord_to_index.items()}

    def _preprocess(self, df):
        sections = []
        chords_tokenized = []
        genres = []
        for index, row in df.iterrows():
            chords_str = row['chords']
            main_genre = row['main_genre']
            # print(f"\nProcesando chords_str: '{chords_str}'")
            split_chords = chords_str.split('<')
            # print(f"Resultado de split('<'): {split_chords}")
            for item in split_chords:
                if '>' in item:
                    section_label, chord_sequence = item.split('>', 1)
                    cleaned_label = section_label.strip()
                    sections.append(cleaned_label)
                    chords_tokenized.append([chord.strip() for chord in chord_sequence.strip().split()])
                    genres.append(main_genre)
                    # print(f"  - Item: '{item}', Etiqueta extraída: '{cleaned_label}'")
        return {'sections': sections, 'chords_tokenized': chords_tokenized, 'genres': genres}

    def _build_vocab(self, token_lists):
        tokens = []
        for token_list in token_lists:
            tokens.extend(token_list)
        return sorted(list(set(tokens)))

    def __len__(self):
        return len(self.data['sections'])

    def __getitem__(self, idx):
        section = self.data['sections'][idx]
        chords = self.data['chords_tokenized'][idx]
        genre = self.data['genres'][idx]

        # print(f"Intentando acceder a la sección: '{section}'")
        if section not in self.section_to_index:
            print(f"¡¡¡ERROR!!! La sección '{section}' no está en self.section_to_index: {self.section_to_index.keys()}")

        section_index = self.section_to_index[section]
        chord_indices = [self.chord_to_index[chord] for chord in chords]
        genre_index = self.genre_to_index[genre]

        return {
            'section': torch.tensor(section_index, dtype=torch.long),
            'chords': torch.tensor(chord_indices[:-1], dtype=torch.long),
            'next_chord': torch.tensor(chord_indices[1:], dtype=torch.long),
            'genre': torch.tensor(genre_index, dtype=torch.long)
        }
    
def pad_sequences(batch):
    sections = [item['section'] for item in batch]
    genres = [item['genre'] for item in batch]
    chords = [item['chords'] for item in batch]
    next_chords = [item['next_chord'] for item in batch]

    chords_padded = torch.nn.utils.rnn.pad_sequence(chords, batch_first=True)
    next_chords_padded = torch.nn.utils.rnn.pad_sequence(next_chords, batch_first=True)

    return {
        'section': torch.stack(sections),
        'chords': chords_padded,
        'next_chord': next_chords_padded,
        'genre': torch.stack(genres)
    }

def create_dataloader(df, batch_size=32, shuffle=True):
    dataset = ChordDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_sequences)
    return dataloader, dataset.chord_vocab, dataset.section_vocab, dataset.genre_vocab, dataset.index_to_chord

dataloader, chord_vocab, section_vocab, genre_vocab, index_to_chord = create_dataloader(df, batch_size=batch_size, shuffle=True)

# --- Definició del model ---
class ChordGenerator(nn.Module):
    def __init__(self, chord_vocab_size, section_vocab_size, genre_vocab_size, embedding_dim, hidden_dim, num_layers):
        super(ChordGenerator, self).__init__()
        self.chord_embedding = nn.Embedding(chord_vocab_size, embedding_dim)
        self.section_embedding = nn.Embedding(section_vocab_size, embedding_dim)
        self.genre_embedding = nn.Embedding(genre_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 3, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, chord_vocab_size)

    def forward(self, chords, section, genre):
        chord_embedded = self.chord_embedding(chords)
        section_embedded = self.section_embedding(section).unsqueeze(1).expand(-1, chords.size(1), -1)
        genre_embedded = self.genre_embedding(genre).unsqueeze(1).expand(-1, chords.size(1), -1)
        embedded = torch.cat((chord_embedded, section_embedded, genre_embedded), dim=2)
        output, _ = self.lstm(embedded)
        prediction = self.linear(output)
        return prediction
model = ChordGenerator(len(chord_vocab), len(section_vocab), len(genre_vocab), embedding_dim, hidden_dim, num_layers)
model.load_state_dict(torch.load("../chord_generator.pth"))

# --- Generació d'acords ---
def generate_chords(model, start_sequence, section_label, genre_label, chord_to_index, section_to_index, genre_to_index, index_to_chord, max_length=50, device="cpu"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        start_indices = [chord_to_index[chord] for chord in start_sequence]
        input_sequence = torch.tensor(start_indices, dtype=torch.long).unsqueeze(0).to(device)
        section_index = torch.tensor([section_to_index[section_label]], dtype=torch.long).to(device)
        genre_index = torch.tensor([genre_to_index[genre_label]], dtype=torch.long).to(device)

        generated_sequence = start_indices[:]

        for _ in range(max_length):
            outputs = model(input_sequence, section_index, genre_index)
            probabilities = torch.softmax(outputs[:, -1, :], dim=-1)
            next_chord_index = torch.multinomial(probabilities, num_samples=1).item()
            generated_sequence.append(next_chord_index)
            input_sequence = torch.cat((input_sequence, torch.tensor([[next_chord_index]], dtype=torch.long).to(device)), dim=1)

        return [index_to_chord[idx] for idx in generated_sequence]
