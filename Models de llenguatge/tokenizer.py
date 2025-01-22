import unicodedata
from collections import Counter

def normalitzar_char(char):
    char = char.lower() # passem a minúscula
    char = unicodedata.normalize("NFD", char)
    char = ''.join(c for c in char if unicodedata.category(c) != 'Mn' and not c.isspace()) # eliminem els accents del caràcter i els espais en blanc
    return char

def construir_diccionari():
    diccionari = {}

    for i in range(10): # Afegim els números del 0 al 9
        diccionari[str(i)] = i

    for i, char in enumerate("abcdefghijklmnñopqrstuvwxyz", start=10): # Afegim lletres de la a-z
        diccionari[char] = i

    for i, char in enumerate(".,!?", start=37): # caràcters especials
        diccionari[char] = i

    return diccionari

diccionari = construir_diccionari()

def trobar_sequencies(tokens, longitud, umbral):
    sequencies = Counter(tuple(tokens[i:i+longitud]) for i in range(len(tokens) - longitud + 1)) # contem quants cops es repeteixen les parelles de tokens
    frequents = {seq: count for seq, count in sequencies.items() if count >= umbral} # ens quedem amb les sequencies que es repeteixin mes cops que l'umbral indicat
    return frequents

def actualitzar_diccionari(diccionari, frequents, index_inici):
    nou_diccionari = diccionari.copy() # copiem el diccionari actual
    nou_index = index_inici
    for seq in frequents:
        if seq not in nou_diccionari: # si la sequencia no es troba al diccionari l'afegim
            nou_diccionari[seq] = nou_index
            nou_index += 1
    return nou_diccionari

def substituir_sequencies(tokens, frequents, diccionari):
    i = 0
    resultat = []
    while i < len(tokens):
        substituit = False
        for longitut in range(max(len(seq) for seq in frequents), 0, -1):
            if i + longitut <= len(tokens):
                subsequencia = tuple(tokens[i:i+longitut])
                if subsequencia in frequents:
                    resultat.append(diccionari[subsequencia])
                    i += longitut
                    substituit = True
                    break
        if not substituit:
            resultat.append(tokens[i])
            i += 1
    return resultat

def tokenitzar_loop(tokens, diccionari, num_tokens, longitut=2, umbral=3, index_inici=41):
    diccionari_actualitzat = diccionari.copy()
    tokens_actualitzats = tokens
    index_actual = index_inici

    while len(tokens_actualitzats) > num_tokens:
        sequencies_frequents = trobar_sequencies(tokens_actualitzats, longitut, umbral)
        
        if not sequencies_frequents:
            break

        diccionari_actualitzat = actualitzar_diccionari(diccionari_actualitzat, sequencies_frequents, index_actual)

        tokens_actualitzats = substituir_sequencies(tokens_actualitzats, sequencies_frequents, diccionari_actualitzat)

        index_actual = max(diccionari_actualitzat.values()) + 1

    return tokens_actualitzats, diccionari_actualitzat

with open("data/dracula_test.txt", "r", encoding="utf-8") as file: # Carregar el text, en aquest cas serà la novel·la de Dràcula
    text = file.read()

text_normalitzat = [normalitzar_char(char) for char in text if normalitzar_char(char)] # Normalitzem els caràcters de la novel·la i eliminem els espais en blanc

tokens = [diccionari[char] for char in text_normalitzat if char in diccionari] # Passem el text normalitzat al nostre codi ascii que hem indicat al diccionari = tokens

tokens_finals, diccionari_final = tokenitzar_loop(tokens, diccionari, num_tokens=100)

print(diccionari_final)