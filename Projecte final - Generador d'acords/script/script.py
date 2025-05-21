import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ChordDataset, ChordGenerator, generate_chords, chord_vocab, section_vocab, genre_vocab

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
print(f"Using device: {device}")

try:
    dummy_dataset = ChordDataset(df)
    print("Dataset obtingut correctament!")
except Exception as e:
    print(f"Error al inicialitzar ChordDataset: {e}")
    exit()

# Instància del model
model = ChordGenerator(len(chord_vocab), len(section_vocab), len(genre_vocab), embedding_dim, hidden_dim, num_layers)

# Ruta on es troba el model
model_path = "../chord_generator.pth"

if not os.path.exists(model_path):
    print(f"Error: El archivo del modelo '{model_path}' no se encontró. Asegúrate de que el modelo esté en la misma carpeta o especifica la ruta completa.")
    exit()

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Modelo '{model_path}' carregat correctament.")
except Exception as e:
    print(f"Ha ocorregut un error al carregar el model: {e}")
    exit()

def run_interactive_generation():
    print("\n--- Generador Interactiu d'Acords ---")
    print("Introdueix els paràmetres per a la generació.")

    print(f"\nAcords disponibles (alguns): {list(dummy_dataset.chord_to_index.keys())[:10]}...")
    print(f"Seccions disponibles: {list(dummy_dataset.section_to_index.keys())}")
    print(f"Gèneres disponibles: {list(dummy_dataset.genre_to_index.keys())}")


    while True:
        try:
            start_seq_str = input("\nIntrodueix la seqüència inicial d'acords (separats per espai, ex: C G Am): ").strip()
            start_sequence = start_seq_str.split()

            section_label = input("Introdueix la secció (ex: verse, chorus, bridge): ").strip().lower()
            if section_label not in dummy_dataset.section_to_index:
                print(f"Error: La secció '{section_label}' no és vàlida. Si us plau, tria'n una de les disponibles.")
                continue

            genre_label = input("Introdueix el gènere (ex: pop, rock, metal): ").strip().lower()
            if genre_label not in dummy_dataset.genre_to_index:
                print(f"Error: El gènere '{genre_label}' no és vàlid. Si us plau, tria'n un de les disponibles.")
                continue

            max_len_str = input("Longitud màxima de la seqüència d'acords a generar (ex: 15): ")
            max_length = int(max_len_str)
            if max_length <= 0:
                print("La longitud màxima ha de ser un nombre positiu.")
                continue

            generated_chords = generate_chords(
                model,
                start_sequence,
                section_label,
                genre_label,
                dummy_dataset.chord_to_index,
                dummy_dataset.section_to_index,
                dummy_dataset.genre_to_index,
                dummy_dataset.index_to_chord,
                device=device,
                max_length=max_length
            )

            if generated_chords:
                print(f"\n--- Acords Generats ---")
                print(f"Inici: {start_sequence}, Secció: {section_label}, Gènere: {genre_label}")
                print(f"Seqüència: {' '.join(generated_chords)}")
            else:
                print("No s'han pogut generar acords amb l'entrada proporcionada.")

            another_go = input("\nVols generar una altra seqüència? (s/n): ").strip().lower()
            if another_go != 's':
                break

        except ValueError:
            print("Entrada no vàlida. Si us plau, introdueix un nombre per a la longitud.")
        except Exception as e:
            print(f"Ha ocorregut un error inesperat: {e}")

if __name__ == "__main__":
    run_interactive_generation()