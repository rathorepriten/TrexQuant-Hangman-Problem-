import torch
import math
import numpy as np
import string
import re
import collections
import random

"""
Hangman Solver - BiLSTM Neural Network Inference Script

This script provides the inference functionality for the Hangman solver,
using a trained bidirectional LSTM model to predict the most likely letters
for partially revealed words. It includes advanced context-aware letter selection
strategies and fallback mechanisms for robust performance.
"""

# Improved BiLSTM model with positional encoding and pretrained embeddings
class ImprovedHangmanPredictor(torch.nn.Module):
    """
    Bidirectional LSTM model for predicting missing characters in partially revealed words.
    
    This model combines character embeddings (either pretrained or learned) with positional
    encodings, and processes them through bidirectional LSTM layers to predict the most
    likely character at each masked position.
    
    Features:
    - Character embeddings (optionally pretrained)
    - Positional encoding for character position awareness
    - Multi-layer BiLSTM for context capturing
    - Dropout regularization to prevent overfitting
    - Additional FC layers for improved expressiveness
    """
    def __init__(self, input_size, hidden_size, output_size, max_seq_length=20, 
                 embedding_dim=100, pretrained_embeddings=None, num_layers=3, dropout=0.2):
        """
        Initialize the Hangman prediction model.
        
        Args:
            input_size (int): Size of the character vocabulary
            hidden_size (int): Size of the hidden state in the LSTM
            output_size (int): Size of the output (usually same as input_size)
            max_seq_length (int): Maximum word length to support
            embedding_dim (int): Dimension of character embeddings
            pretrained_embeddings (numpy.ndarray, optional): Pretrained character embeddings
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
        """
        super(ImprovedHangmanPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Use pretrained embeddings if available
        if pretrained_embeddings is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False,
                padding_idx=0
            )
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            # Standard embedding layer
            self.embedding = torch.nn.Embedding(input_size, embedding_dim)
        
        # Embedding projection layer
        self.embedding_projection = None
        if embedding_dim != hidden_size:
            self.embedding_projection = torch.nn.Linear(embedding_dim, hidden_size)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, hidden_size)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)
        
        # Bi-LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Additional fully connected layers
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.activation = torch.nn.ReLU()
        
        # Output layer
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
        # Softmax for output probabilities
        self.softmax = torch.nn.LogSoftmax(dim=2)
    
    def _create_positional_encoding(self, max_seq_length, d_model):
        """
        Create positional encoding matrix for sequence positions.
        
        This uses the sine/cosine encoding scheme similar to that in the Transformer
        architecture, which helps the model understand the relative positions of
        characters within a word.
        
        Args:
            max_seq_length (int): Maximum sequence length
            d_model (int): Model dimension size
            
        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, max_seq_length, d_model)
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-(d_model//2)])
            
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x, lengths=None):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of character indices, shape (batch_size, seq_length)
            lengths (torch.Tensor, optional): Lengths of sequences in the batch
            
        Returns:
            torch.Tensor: Log probabilities for each character at each position
        """
        # x shape: (batch_size, seq_length)
        batch_size, seq_length = x.size()
        
        # Convert input to embeddings
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Project embeddings if needed
        if self.embedding_projection is not None:
            embedded = self.embedding_projection(embedded)
        
        # Add positional encoding
        positional_encoding = self.positional_encoding[:, :seq_length, :].to(embedded.device)
        embedded = embedded + positional_encoding
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        if lengths is not None:
            # Pack padded sequence
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Pass through LSTM
            packed_output, _ = self.lstm(packed)
            
            # Unpack sequence
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            # If lengths not provided, process normally
            output, _ = self.lstm(embedded)
        
        # Pass through additional FC layer with activation
        output = self.fc1(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        # Pass through final layer and apply softmax
        output = self.fc2(output)
        output = self.softmax(output)
        
        return output


# Letter frequency distribution in English (fallback)
ENGLISH_LETTER_FREQ = {
    'e': 12.02, 't': 9.10, 'a': 8.12, 'o': 7.68, 'i': 7.31, 'n': 6.95, 's': 6.28,
    'r': 6.02, 'h': 5.92, 'd': 4.32, 'l': 3.98, 'u': 2.88, 'c': 2.71, 'm': 2.61,
    'f': 2.30, 'y': 2.11, 'w': 2.09, 'g': 2.03, 'p': 1.82, 'b': 1.49, 'v': 1.11,
    'k': 0.69, 'x': 0.17, 'q': 0.11, 'j': 0.10, 'z': 0.07
}

# Position-specific letter frequencies
POSITION_LETTER_FREQ = {
    'start': {  # First letter frequencies
        't': 16.0, 's': 11.3, 'a': 8.5, 'c': 8.1, 'p': 7.1, 'b': 6.0, 'm': 5.8,
        'd': 5.7, 'f': 4.2, 'r': 4.1, 'h': 3.7, 'e': 3.6, 'w': 3.6, 'l': 3.1,
        'g': 3.0, 'i': 1.8, 'n': 1.2, 'o': 1.1, 'v': 1.0, 'j': 0.7, 'k': 0.6,
        'q': 0.3, 'u': 0.3, 'y': 0.2, 'z': 0.1, 'x': 0.0
    },
    'end': {  # Last letter frequencies
        'e': 16.8, 's': 14.0, 't': 8.7, 'd': 8.5, 'n': 7.6, 'y': 6.7, 'r': 6.6,
        'a': 4.8, 'l': 4.8, 'o': 4.1, 'h': 3.0, 'g': 2.2, 'k': 1.9, 'm': 1.8,
        'p': 1.8, 'c': 1.3, 'i': 1.2, 'f': 1.1, 'b': 0.9, 'u': 0.9, 'w': 0.8,
        'z': 0.2, 'x': 0.2, 'v': 0.1, 'j': 0.0, 'q': 0.0
    }
}

# Common letter patterns in English
COMMON_PATTERNS = {
    'q': {'u': 0.95},  # q is almost always followed by u
    'th': {'e': 0.5, 'a': 0.2, 'i': 0.15, 'o': 0.1},
    'ch': {'e': 0.3, 'a': 0.2, 'i': 0.15, 'o': 0.15},
    'sh': {'e': 0.3, 'a': 0.2, 'i': 0.15, 'o': 0.15},
    'ing': {'e': 0.3, 's': 0.2, 'l': 0.1},
    'tion': {'a': 0.3, 's': 0.3, 'e': 0.2},
}

def load_model(model_path='hangman_bilstm_model.pth'):
    """
    Load the trained Hangman model.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        tuple: (model, char_to_idx, idx_to_char) or (None, None, None) on error
    """
    try:
        # Load on CPU to ensure compatibility
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # Get model parameters
        input_size = len(char_to_idx)
        hidden_size = 256  # Match your model's hidden size
        output_size = len(char_to_idx)
        max_seq_length = 40
        embedding_dim = 100
        
        # Create model with same architecture
        model = ImprovedHangmanPredictor(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size,
            max_seq_length=max_seq_length,
            embedding_dim=embedding_dim,
            num_layers=3,
            dropout=0.2
        )
        
        # Load the weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        return model, char_to_idx, idx_to_char
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Initialize the model, character mappings, and device
_model = None
_char_to_idx = None
_idx_to_char = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_guess(word, guessed_letters):
    """
    Use trained model to guess next letter for Hangman.
    
    This function takes the current state of a Hangman game and uses the trained
    BiLSTM model to predict the most likely letter for the next guess, using
    context-aware position weighting and linguistic knowledge.
    
    Args:
        word: Current state of the word with underscores for unknown letters (e.g., "_ p p _ e")
        guessed_letters: Set of letters that have already been guessed
        
    Returns:
        str: A letter to guess
    """
    global _model, _char_to_idx, _idx_to_char, _device
    
    # Load model if not already loaded
    if _model is None:
        _model, _char_to_idx, _idx_to_char = load_model()
        if _model is not None:
            _model = _model.to(_device)
    
    # Clean up the word (remove spaces)
    clean_word = word.replace(" ", "")
    word_len = len(clean_word)
    
    # If model failed to load, fall back to letter frequency
    if _model is None or _char_to_idx is None or _idx_to_char is None:
        # Use letter frequency fallback
        remaining_letters = [l for l in string.ascii_lowercase if l not in guessed_letters]
        letter_freqs = {l: ENGLISH_LETTER_FREQ.get(l, 0.01) for l in remaining_letters}
        return max(letter_freqs.items(), key=lambda x: x[1])[0]
    
    # Convert word to model input
    input_indices = [_char_to_idx.get(c.lower(), _char_to_idx['_']) for c in clean_word]
    input_tensor = torch.tensor([input_indices]).to(_device)
    length_tensor = torch.tensor([word_len]).to(_device)
    
    # Make prediction
    with torch.no_grad():
        output = _model(input_tensor, length_tensor)
    
    # Get probabilities across all positions
    pos_probs = output[0, :word_len].exp().cpu().numpy()
    
    # Find masked positions
    masked_positions = [i for i, c in enumerate(clean_word) if c == '_']
    
    if not masked_positions:
        # No masked positions left (shouldn't happen in normal play)
        return random.choice([l for l in string.ascii_lowercase if l not in guessed_letters])
    
    # Context-aware weighting for masked positions
    masked_probs = []
    
    for pos in masked_positions:
        # Default weight
        weight = 1.0
        
        # Check for special patterns
        left_context = ""
        if pos > 0:
            # Get left context (up to 3 characters)
            for i in range(max(0, pos-3), pos):
                if clean_word[i] != '_':
                    left_context += clean_word[i]
            
            # Increase weight if adjacent to a revealed letter
            if pos > 0 and clean_word[pos-1] != '_':
                weight += 0.5
                
                # Special case: q is almost always followed by u
                if clean_word[pos-1] == 'q':
                    # Extremely high weight for 'u' after 'q'
                    pos_probs[pos, _char_to_idx.get('u', 0)] *= 50.0
        
        # Check right context
        right_context = ""
        if pos < len(clean_word)-1:
            # Get right context (up to 3 characters)
            for i in range(pos+1, min(len(clean_word), pos+4)):
                if clean_word[i] != '_':
                    right_context += clean_word[i]
            
            # Increase weight if adjacent to a revealed letter
            if clean_word[pos+1] != '_':
                weight += 0.5
            
        # Extra weight for isolated gaps (highly constrained)
        is_isolated = (pos > 0 and clean_word[pos-1] != '_' and 
                      pos < len(clean_word)-1 and clean_word[pos+1] != '_')
        if is_isolated:
            weight += 1.0
            
        # Position-specific weighting
        if pos == 0:  # First letter
            weight += 0.25
            # Boost common first letters
            for letter, freq in POSITION_LETTER_FREQ['start'].items():
                letter_idx = _char_to_idx.get(letter, None)
                if letter_idx is not None:
                    pos_probs[pos, letter_idx] *= (1.0 + freq/100.0)
        elif pos == len(clean_word) - 1:  # Last letter
            weight += 0.25
            # Boost common last letters
            for letter, freq in POSITION_LETTER_FREQ['end'].items():
                letter_idx = _char_to_idx.get(letter, None)
                if letter_idx is not None:
                    pos_probs[pos, letter_idx] *= (1.0 + freq/100.0)
            
        # Apply weights
        masked_probs.append(pos_probs[pos] * weight)
    
    # Sum the weighted probabilities
    if masked_probs:
        summed_probs = np.sum(masked_probs, axis=0)
    else:
        # Fallback (should never happen)
        masked_probs = pos_probs[masked_positions]
        summed_probs = np.sum(masked_probs, axis=0)
    
    # Filter out already guessed letters
    letter_probs = {}
    for idx, prob in enumerate(summed_probs):
        char = _idx_to_char.get(idx)
        if char and char != '_' and char.isalpha() and char not in guessed_letters:
            letter_probs[char] = prob
    
    # If no valid predictions, use letter frequency
    if not letter_probs:
        remaining_letters = [l for l in string.ascii_lowercase if l not in guessed_letters]
        if not remaining_letters:
            # No letters left to guess - shouldn't happen in normal play
            return random.choice(string.ascii_lowercase)
        
        # Use position-specific letter frequencies if applicable
        if len(masked_positions) == 1:
            pos = masked_positions[0]
            if pos == 0:
                freq_dict = POSITION_LETTER_FREQ['start']
            elif pos == len(clean_word) - 1:
                freq_dict = POSITION_LETTER_FREQ['end']
            else:
                freq_dict = ENGLISH_LETTER_FREQ
        else:
            freq_dict = ENGLISH_LETTER_FREQ
            
        letter_freqs = {l: freq_dict.get(l, 0.01) for l in remaining_letters}
        return max(letter_freqs.items(), key=lambda x: x[1])[0]
    
    # Return the letter with highest probability
    return max(letter_probs.items(), key=lambda x: x[1])[0]

# Module-level initialization
def initialize_model():
    """
    Initialize the model when the module is loaded.
    
    This function loads the trained model into memory to be ready for inference.
    """
    global _model, _char_to_idx, _idx_to_char, _device
    _model, _char_to_idx, _idx_to_char = load_model()
    if _model is not None:
        _model = _model.to(_device)
        print("Hangman model loaded successfully!")
    else:
        print("Warning: Failed to load Hangman model. Will use fallback methods.")

# Call initialization
initialize_model()

def guess(self, word):
    """
    Advanced guess function for the Hangman API.
    
    This function is designed to be called by the Hangman API. It takes the current state
    of the word and returns the best letter to guess next, using the trained model
    and fallback mechanisms.
    
    Args:
        word (str): Current state of the word with underscores (e.g., "_ p p _ e")
        
    Returns:
        str: A single letter as your guess
    """
    # Get the letter from the model
    guess_letter = model_guess(word, set(self.guessed_letters))
    
    # Fallback to dictionary approach if model gives an already guessed letter
    # or if the model failed to load
    if guess_letter in self.guessed_letters:
        # Clean the word for dictionary matching
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        
        # Find matching words from dictionary
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        for dict_word in current_dictionary:
            if len(dict_word) != len_word:
                continue
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)
        
        self.current_dictionary = new_dictionary
        
        # Count character frequencies in matching words
        full_dict_string = "".join(new_dictionary)
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()
        
        # Get the most frequent letter that hasn't been guessed
        guess_letter = '!'
        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
        
        # If no matches in dictionary, use global letter frequency
        if guess_letter == '!' or guess_letter in self.guessed_letters:
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break
    
    return guess_letter

def play_hangman_game(secret_word, max_guesses=6):
    """
    Play a complete Hangman game for evaluation and demonstration.
    
    This function simulates playing a full Hangman game using the trained model.
    
    Args:
        secret_word (str): The word to guess
        max_guesses (int): Maximum number of incorrect guesses allowed
        
    Returns:
        dict: Game result details including win/loss, guesses made, etc.
    """
    word = secret_word.lower()
    word_len = len(word)
    
    # Initialize game state
    current_state = "_" * word_len
    guessed_letters = set()
    incorrect_guesses = 0
    
    # Play until win or max guesses reached
    while "_" in current_state and incorrect_guesses < max_guesses:
        # Show current game state
        display_word = " ".join(current_state)
        print(f"Word: {display_word}")
        print(f"Guessed letters: {', '.join(sorted(guessed_letters))}")
        print(f"Incorrect guesses: {incorrect_guesses}/{max_guesses}")
        
        # Get next guess
        guess_letter = model_guess(current_state, guessed_letters)
        guessed_letters.add(guess_letter)
        print(f"Guessing: {guess_letter}")
        
        # Check if correct
        if guess_letter in word:
            # Update the current state
            new_state = ""
            for i, letter in enumerate(word):
                if letter == guess_letter or current_state[i] != "_":
                    new_state += letter
                else:
                    new_state += "_"
            current_state = new_state
            print(f"Correct! '{guess_letter}' is in the word.")
        else:
            incorrect_guesses += 1
            print(f"Wrong! '{guess_letter}' is not in the word.")
        
        print("-" * 40)
    
    # Game over - show result
    if "_" not in current_state:
        print(f"You won! The word was '{word}'")
        won = True
    else:
        print(f"You lost! The word was '{word}'")
        won = False
    
    return {
        'word': word,
        'won': won,
        'guessed_letters': guessed_letters,
        'incorrect_guesses': incorrect_guesses,
        'final_state': current_state
    }
