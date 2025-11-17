import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm 
import math
import os
import gzip
import shutil
import urllib.request
import io

"""
Hangman Solver - BiLSTM Neural Network Training Script

This script trains a bidirectional LSTM neural network to predict missing letters
in partially revealed words for solving Hangman puzzles. It uses positional encoding
and FastText character embeddings to improve prediction accuracy.

The model is trained using a masked character modeling approach on a large dictionary
of English words, and evaluated by simulating real Hangman games.
"""

def load_fasttext_embeddings(vocab, embedding_dim=100):
    """
    Download and load pretrained FastText character n-gram embeddings.
    
    These embeddings capture subword information which is useful for character-level tasks.
    The function will attempt to download the embeddings if they don't exist locally,
    and will create a mapping between our character vocabulary and the embedding space.
    
    Args:
        vocab (dict): Dictionary mapping characters to indices
        embedding_dim (int): Dimension of the embeddings to use
        
    Returns:
        numpy.ndarray: Matrix of embeddings for each character in our vocabulary,
                       or None if embeddings couldn't be loaded
    """
    print("Loading fastText character n-gram embeddings...")
    
    # URL for a smaller English fastText embeddings
    fasttext_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
    
    # Local paths
    zip_path = "fasttext.zip"
    vec_path = "crawl-300d-2M-subword.vec"
    
    # Check if the embeddings file already exists
    if not os.path.exists(vec_path):
        print(f"Downloading fastText embeddings from {fasttext_url}...")
        
        # Download and extract
        try:
            import zipfile
            urllib.request.urlretrieve(fasttext_url, zip_path)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
                
            # Clean up
            os.remove(zip_path)
            print("Downloaded and extracted fastText embeddings.")
        except Exception as e:
            print(f"Error downloading fastText embeddings: {e}")
            print("Using random initialization instead.")
            return None
    
    # Load the embeddings
    try:
        # Initialize embedding dictionary
        embeddings_dict = {}
        
        # Read embeddings file
        with open(vec_path, 'r', encoding='utf-8') as f:
            # Skip header
            header = next(f)
            print(f"FastText header: {header}")
            
            # Read line by line
            for line in tqdm(f, desc="Loading fastText vectors"):
                values = line.strip().split(' ')
                word = values[0]
                vector = np.array([float(val) for val in values[1:]])
                
                # Resize vector to match embedding_dim
                if len(vector) > embedding_dim:
                    vector = vector[:embedding_dim]  # Truncate if too large
                elif len(vector) < embedding_dim:
                    # Pad with zeros if too small
                    pad_size = embedding_dim - len(vector)
                    vector = np.pad(vector, (0, pad_size), 'constant')
                    
                embeddings_dict[word] = vector
                
        print(f"Loaded {len(embeddings_dict)} word vectors.")
        
        # Create embedding matrix for our vocabulary
        embedding_matrix = np.zeros((len(vocab), embedding_dim))
        found_count = 0
        
        # Try to find embeddings for each character in our vocabulary
        for char, idx in vocab.items():
            # Skip the mask token
            if char == '_':
                continue
                
            # Try exact match first
            if char in embeddings_dict:
                embedding_matrix[idx] = embeddings_dict[char]
                found_count += 1
            else:
                # If character not found, try to find similar subwords
                char_embedding = np.zeros(embedding_dim)
                found_similar = False
                similar_count = 0
                
                # Check for subwords containing this character
                # Limit to first 1000 words to avoid excessive computation
                for word, vec in list(embeddings_dict.items())[:1000]:
                    if char in word:
                        char_embedding += vec
                        found_similar = True
                        similar_count += 1
                        # Stop after finding a few examples to save time
                        if similar_count >= 10:
                            break
                        
                if found_similar:
                    embedding_matrix[idx] = char_embedding / np.linalg.norm(char_embedding)
                    found_count += 1
                else:
                    # Random initialization for unknown characters
                    embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
                    
        print(f"Found embeddings for {found_count}/{len(vocab)-1} characters in vocabulary.")
        return embedding_matrix
        
    except Exception as e:
        print(f"Error loading fastText embeddings: {e}")
        print("Using random initialization instead.")
        return None


def create_synthetic_char_embeddings(vocab, embedding_dim=100):
    """
    Create synthetic character embeddings based on character properties when FastText embeddings are unavailable.
    
    This function generates meaningful character embeddings by encoding properties like:
    - Whether the character is a vowel or consonant
    - The character's position in the alphabet
    - Case information (uppercase/lowercase)
    - Common n-gram patterns
    
    Args:
        vocab (dict): Dictionary mapping characters to indices
        embedding_dim (int): Dimension of the embeddings to create
        
    Returns:
        numpy.ndarray: Matrix of synthetic embeddings for each character in the vocabulary
    """
    print("Creating synthetic character embeddings...")
    
    # Initialize embedding matrix
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    
    # For each character in our vocabulary
    for char, idx in vocab.items():
        if char == '_':  # Special mask token
            # Initialize with zeros (or small random values)
            embedding_matrix[idx] = np.random.normal(0, 0.01, embedding_dim)
        else:
            # Create a positionally-encoded embedding
            embedding = np.zeros(embedding_dim)
            
            # Character ASCII value as a base pattern
            char_val = ord(char)
            
            # Set specific dimensions based on character properties
            # Letters at beginning/middle/end of alphabet get different patterns
            if char.isalpha():
                alphabet_pos = (ord(char.lower()) - ord('a')) / 26.0  # 0 to 1
                
                # Set different regions of the embedding based on position in alphabet
                region_size = embedding_dim // 4
                
                # First region: vowel vs consonant pattern
                if char.lower() in 'aeiou':
                    embedding[:region_size] = np.sin(np.arange(region_size) * (alphabet_pos + 0.5))
                else:
                    embedding[:region_size] = np.cos(np.arange(region_size) * (alphabet_pos + 0.5))
                
                # Second region: position in alphabet
                start_idx = region_size
                embedding[start_idx:start_idx+region_size] = np.linspace(0, alphabet_pos, region_size)
                
                # Third region: uppercase vs lowercase
                start_idx = 2 * region_size
                if char.isupper():
                    embedding[start_idx:start_idx+region_size] = np.sin(np.arange(region_size) * 0.1)
                else:
                    embedding[start_idx:start_idx+region_size] = np.cos(np.arange(region_size) * 0.1)
                
                # Fourth region: common n-gram patterns
                start_idx = 3 * region_size
                end_idx = min(4 * region_size, embedding_dim)
                embedding[start_idx:end_idx] = np.random.normal(0, 0.1, end_idx - start_idx)
                
            else:  # Non-alphabetic characters
                embedding = np.random.normal(0, 0.1, embedding_dim)
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embedding_matrix[idx] = embedding
            
    return embedding_matrix


class ImprovedHangmanPredictor(nn.Module):
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
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False,  # Allow fine-tuning
                padding_idx=0  # For the mask token
            )
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            # Standard embedding layer
            self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # Embedding projection layer (if dimensions don't match)
        self.embedding_projection = None
        if embedding_dim != hidden_size:
            self.embedding_projection = nn.Linear(embedding_dim, hidden_size)
        
        # Positional encoding to capture position information
        self.positional_encoding = self._create_positional_encoding(max_seq_length, hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 because of bidirectional
        self.activation = nn.ReLU()
        
        # Output layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Softmax for output probabilities
        self.softmax = nn.LogSoftmax(dim=2)
    
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
        if d_model % 2 == 0:  # Handle both even and odd dimensions
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-(d_model//2)])
            
        return pe.unsqueeze(0)  # Add batch dimension (1, max_seq_length, d_model)
    
    def forward(self, x, lengths):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of character indices, shape (batch_size, seq_length)
            lengths (torch.Tensor): Lengths of sequences in the batch
            
        Returns:
            torch.Tensor: Log probabilities for each character at each position
        """
        # x shape: (batch_size, seq_length)
        batch_size, seq_length = x.size()
        
        # Convert input to embeddings
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Project embeddings if needed
        if self.embedding_projection is not None:
            embedded = self.embedding_projection(embedded)  # (batch_size, seq_length, hidden_size)
        
        # Add positional encoding
        positional_encoding = self.positional_encoding[:, :seq_length, :].to(embedded.device)
        embedded = embedded + positional_encoding
        
        # Apply dropout for regularization
        embedded = self.dropout(embedded)
        
        # Pack padded sequence for more efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_output, _ = self.lstm(packed)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        

        output = self.fc1(output)
        output = self.activation(output)
        output = self.dropout(output)  # Apply dropout again
        

        output = self.fc2(output)  # (batch_size, seq_length, output_size)
        
        # Apply softmax
        output = self.softmax(output)  # (batch_size, seq_length, output_size)
        
        return output


class MaskedWordDataset(Dataset):
    """
    Dataset for masked character modeling to train the Hangman predictor.
    
    This dataset takes a list of words and creates training samples by randomly
    masking some characters, then setting up the task of predicting the original
    characters at those masked positions.
    """
    def __init__(self, words, char_to_idx, mask_prob=0.3, num_samples_per_word=5):
        """
        Initialize the masked word dataset.
        
        Args:
            words (list): List of words to use for training
            char_to_idx (dict): Mapping from characters to indices
            mask_prob (float): Probability of masking each character
            num_samples_per_word (int): Number of different masked versions to create per word
        """
        self.words = words
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        self.mask_prob = mask_prob
        self.num_samples_per_word = num_samples_per_word
        self.mask_token = '_'
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """
        Create training samples by masking random characters in each word.
        
        Returns:
            list: List of tuples (word, masked_word, mask_positions)
        """
        samples = []
        for word in self.words:
            for _ in range(self.num_samples_per_word):
                # Create a masked version of the word
                masked_word = list(word)
                mask_positions = []
                
                # Choose random positions to mask
                indices = list(range(len(word)))
                random.shuffle(indices)
                num_to_mask = max(1, int(len(word) * self.mask_prob))
                mask_indices = indices[:num_to_mask]
                
                # Apply masking
                for idx in mask_indices:
                    masked_word[idx] = self.mask_token
                    mask_positions.append(idx)
                
                samples.append((word, ''.join(masked_word), mask_positions))
        
        return samples
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (input_tensor, target_tensor, mask_positions_tensor, length)
        """
        word, masked_word, mask_positions = self.samples[idx]
        
        # Convert characters to indices
        input_indices = [self.char_to_idx.get(c, self.char_to_idx[self.mask_token]) for c in masked_word]
        input_tensor = torch.tensor(input_indices, dtype=torch.long)
        
        # Create targets (the original characters at masked positions)
        target_tensor = torch.zeros(len(masked_word), len(self.char_to_idx))
        for pos in mask_positions:
            char_idx = self.char_to_idx[word[pos]]
            target_tensor[pos, char_idx] = 1.0
            
        return input_tensor, target_tensor, torch.tensor(mask_positions, dtype=torch.long), len(masked_word)


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length sequences.
    
    This function sorts the batch by sequence length in descending order and
    pads shorter sequences to match the longest one in the batch.
    
    Args:
        batch (list): Batch of samples from the dataset
        
    Returns:
        tuple: (padded_inputs, padded_targets, mask_positions, lengths)
    """
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: x[3], reverse=True)
    
    # Separate the input, target, and mask_positions
    inputs, targets, mask_positions, lengths = zip(*batch)
    
    # Pad the sequences to the same length
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    # Convert lengths to tensor
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_inputs, padded_targets, mask_positions, lengths


def simulate_hangman_game(model, word, char_to_idx, idx_to_char, device, max_wrong_guesses=6):
    """
    Simulate a complete Hangman game for a given word.
    
    This function uses the trained model to play a Hangman game on the provided word,
    making guesses until either the word is completely revealed or the maximum number
    of wrong guesses is reached.
    
    Args:
        model (ImprovedHangmanPredictor): The trained model
        word (str): The secret word to guess
        char_to_idx (dict): Mapping from characters to indices
        idx_to_char (dict): Mapping from indices to characters
        device (torch.device): Device to run the model on
        max_wrong_guesses (int): Maximum number of incorrect guesses allowed
        
    Returns:
        dict: Result of the game including success status, guesses made, etc.
    """
    word = word.lower()
    word_len = len(word)
    
    # Initialize game state
    current_state = ['_'] * word_len
    tried_letters = set()
    wrong_guesses = 0
    game_over = False
    win = False
    
    # Play until game is over
    while not game_over:
        # Convert current state to string
        current_pattern = ' '.join(current_state)
        
        # Get model prediction for next letter
        input_indices = [char_to_idx.get(c.lower(), char_to_idx['_']) for c in current_state]
        input_tensor = torch.tensor([input_indices]).to(device)
        length_tensor = torch.tensor([word_len]).to(device)
        
        with torch.no_grad():
            output = model(input_tensor, length_tensor)
        
        # Sum probabilities across all positions
        pos_probs = output[0, :word_len].exp().cpu().numpy()
        
        # For masked positions, get the predicted letter probabilities
        masked_positions = [i for i, c in enumerate(current_state) if c == '_']
        
        if not masked_positions:
            # All letters guessed correctly
            game_over = True
            win = True
            break
        
        # CONTEXT-AWARE GUESSING:
        # Weight the predictions based on context
        masked_probs = []
        weights = []
        
        for pos in masked_positions:
            # Default weight
            weight = 1.0
            
            # Increase weight if adjacent to a revealed letter (context is more informative)
            if pos > 0 and current_state[pos-1] != '_':
                weight += 0.5
            if pos < len(current_state)-1 and current_state[pos+1] != '_':
                weight += 0.5
                
            # Further increase weight if in an isolated gap (highly constrained)
            is_isolated = (pos > 0 and current_state[pos-1] != '_' and 
                           pos < len(current_state)-1 and current_state[pos+1] != '_')
            if is_isolated:
                weight += 1.0
                
            # Apply the weight
            masked_probs.append(pos_probs[pos] * weight)
            weights.append(weight)
            
        # If we have weights, combine the weighted probabilities
        if masked_probs:
            # Sum the weighted probabilities
            weighted_sum = np.sum(masked_probs, axis=0)
            summed_probs = weighted_sum
        else:
            # Fallback to simple sum if weights couldn't be applied
            masked_probs = pos_probs[masked_positions]
            summed_probs = np.sum(masked_probs, axis=0)
        
        # Filter out already tried letters
        letter_probs = {}
        for idx, prob in enumerate(summed_probs):
            char = idx_to_char.get(idx)
            if char and char != '_' and char.isalpha() and char not in tried_letters:
                letter_probs[char] = prob
        
        # If no valid predictions, use letter frequency
        if not letter_probs:
            remaining_letters = [l for l in string.ascii_lowercase if l not in tried_letters]
            if not remaining_letters:
                game_over = True
                break
            guess_letter = random.choice(remaining_letters)
        else:
            # Get the letter with highest probability
            guess_letter = max(letter_probs.items(), key=lambda x: x[1])[0]
        
        # Record the guess
        tried_letters.add(guess_letter)
        
        # Check if the guess is correct
        if guess_letter in word:
            # Update current state
            for i, letter in enumerate(word):
                if letter == guess_letter:
                    current_state[i] = letter
        else:
            wrong_guesses += 1
            if wrong_guesses >= max_wrong_guesses:
                game_over = True
                break
    
    # Check if all letters were guessed
    win = ''.join(current_state) == word
    
    return {
        'word': word,
        'guessed_state': ''.join(current_state),
        'tried_letters': tried_letters,
        'wrong_guesses': wrong_guesses,
        'win': win
    }


def evaluate_accuracy(model, data_loader, criterion, device):
    """
    Evaluate the model's accuracy on a dataset.
    
    This function computes both the loss and the accuracy (percentage of correct
    character predictions at masked positions) on the given dataset.
    
    Args:
        model (ImprovedHangmanPredictor): The model to evaluate
        data_loader (DataLoader): DataLoader for the evaluation dataset
        criterion: Loss function
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets, mask_positions, lengths in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths)
            
            # Compute loss
            loss = 0
            for i, length in enumerate(lengths):
                seq_loss = criterion(
                    outputs[i, :length].view(length, -1), 
                    targets[i, :length].view(length, -1)
                )
                loss += seq_loss
            
            # Average loss across batch
            batch_size = inputs.size(0)
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
            
            # Compute accuracy for masked positions
            for i, positions in enumerate(mask_positions):
                for pos in positions:
                    pred_idx = torch.argmax(outputs[i, pos]).item()
                    target_idx = torch.argmax(targets[i, pos]).item()
                    correct_predictions += (pred_idx == target_idx)
                    total_predictions += 1
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct_predictions / max(1, total_predictions)
    
    return avg_loss, accuracy


def play_sample_games(model, val_words, char_to_idx, idx_to_char, device, num_samples=1000):
    """
    Play Hangman games on a sample of words for evaluation.
    
    This function selects a random sample of words and plays a Hangman game
    on each one using the trained model, then returns the results.
    
    Args:
        model (ImprovedHangmanPredictor): The trained model
        val_words (list): List of words to sample from
        char_to_idx (dict): Mapping from characters to indices
        idx_to_char (dict): Mapping from indices to characters
        device (torch.device): Device to run the model on
        num_samples (int): Number of words to sample
        
    Returns:
        list: Results of each game
    """
    # Sample random words
    if len(val_words) > num_samples:
        sample_words = random.sample(val_words, num_samples)
    else:
        sample_words = val_words
    
    results = []
    for word in tqdm(sample_words, desc="Playing sample games"):
        result = simulate_hangman_game(model, word, char_to_idx, idx_to_char, device)
        results.append(result)
    
    return results


def load_dictionary(file_path):
    """
    Load a dictionary of words from a text file.
    
    Args:
        file_path (str): Path to the dictionary file
        
    Returns:
        list: List of words
    """
    with open(file_path, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return words


def train_model(model, train_loader, val_loader, criterion, optimizer, device, val_words, char_to_idx, idx_to_char, num_epochs=5):
    """
    Train the Hangman predictor model.
    
    This function trains the model for the specified number of epochs, evaluating
    its performance after each epoch on both the training and validation datasets,
    as well as by playing sample Hangman games. It saves the best model based on
    validation loss.
    
    Args:
        model (ImprovedHangmanPredictor): The model to train
        train_loader (DataLoader): DataLoader for the training dataset
        val_loader (DataLoader): DataLoader for the validation dataset
        criterion: Loss function
        optimizer: Optimizer
        device (torch.device): Device to train on
        val_words (list): List of words for playing sample games
        char_to_idx (dict): Mapping from characters to indices
        idx_to_char (dict): Mapping from indices to characters
        num_epochs (int): Number of epochs to train for
        
    Returns:
        tuple: (trained_model, training_history)
    """
    best_val_loss = float('inf')
    best_model = None
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets, _, lengths in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, lengths)
            
            # Compute loss
            loss = 0
            for i, length in enumerate(lengths):
                seq_loss = criterion(
                    outputs[i, :length].view(length, -1), 
                    targets[i, :length].view(length, -1)
                )
                loss += seq_loss
            
            # Average loss across batch
            loss = loss / inputs.size(0)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Compute training metrics
        print("Evaluation on training data:")
        train_avg_loss, train_accuracy = evaluate_accuracy(model, train_loader, criterion, device)
        
        # Validation
        print("Evaluation on validation data:")
        val_avg_loss, val_accuracy = evaluate_accuracy(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Play sample games
        print("\nPlaying sample Hangman games:")
        game_results = play_sample_games(model, val_words, char_to_idx, idx_to_char, device, num_samples=1000)
        
        wins = sum(1 for game in game_results if game['win'])
        win_rate = wins / len(game_results)
        avg_wrong = sum(game['wrong_guesses'] for game in game_results) / len(game_results)
        
        print(f"Sample games - Win rate: {win_rate:.2f}, Avg wrong guesses: {avg_wrong:.2f}")
        
        # Print details of each game
        for i, game in enumerate(game_results):
            if i==5:  # Just show the first 5 games for cleanliness
                break
            status = "WIN" if game['win'] else "LOSS"
            print(f"Game {i+1}: Word='{game['word']}', Status={status}, " 
                  f"Final State='{game['guessed_state']}', Wrong Guesses={game['wrong_guesses']}")
        
        # Save training history
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_avg_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_avg_loss,
            'val_accuracy': val_accuracy,
            'game_win_rate': win_rate,
            'game_avg_wrong': avg_wrong
        }
        training_history.append(epoch_history)
        
        # Save the best model
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_model = model.state_dict().copy()
            # Save the model
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': char_to_idx,
                'training_history': training_history,
            }, 'hangman_bilstm_model.pth')
            print("New best model saved!")
    
    # Load the best model
    model.load_state_dict(best_model)
    return model, training_history


def main():
    """
    Main function to load data, train the model, and save the results.
    """
    # Load dictionary
    dictionary_path = "words_250000_train.txt"  # Path to training dictionary
    words = load_dictionary(dictionary_path)

    
    print(f"Loaded {len(words)} words from the dictionary.")
    
    # Filter words (keep only words of reasonable length with alphabetic characters)
    words = [word for word in words if 3 <= len(word) <= 15 and all(c.isalpha() for c in word)]
    print(f"After filtering, {len(words)} words remain.")
    
    # Create character vocabulary
    all_chars = set()
    for word in words:
        all_chars.update(word)
    
    char_to_idx = {'_': 0}  # Start with mask token
    for i, char in enumerate(sorted(all_chars), 1):
        char_to_idx[char] = i
    
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # Load fastText embeddings or create synthetic ones
    embedding_dim = 100  # Smaller embedding dimension to reduce memory usage
    
    # Try to load pretrained embeddings
    pretrained_embeddings = load_fasttext_embeddings(char_to_idx, embedding_dim)
    
    # If embeddings couldn't be loaded, create synthetic ones
    if pretrained_embeddings is None:
        pretrained_embeddings = create_synthetic_char_embeddings(char_to_idx, embedding_dim)
        print("Using synthetic character embeddings instead of fastText.")
    
    # Split into train and validation sets
    random.shuffle(words)  # This shuffles the list in-place
    split_idx = int(0.9 * len(words))
    train_words = words[:split_idx]
    val_words = words[split_idx:]
    
    # Create datasets with stratified word length distribution
    train_dataset = MaskedWordDataset(train_words, char_to_idx, mask_prob=0.4)
    val_dataset = MaskedWordDataset(val_words, char_to_idx, mask_prob=0.4)
    
    # Create dataloaders with our custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  # Large batch size for efficiency
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256,
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Define the improved model with positional encoding, pretrained embeddings and multiple layers
    input_size = len(char_to_idx)
    hidden_size = 256
    output_size = len(char_to_idx)
    max_seq_length = 20
    
    model = ImprovedHangmanPredictor(
        input_size=input_size, 
        hidden_size=hidden_size, 
        output_size=output_size,
        max_seq_length=max_seq_length,
        embedding_dim=embedding_dim,
        pretrained_embeddings=pretrained_embeddings,
        num_layers=3,  # 3 BiLSTM layers
        dropout=0.2
    )
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Train the model
    print("Starting training...")
    model, training_history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device,
        val_words,
        char_to_idx,
        idx_to_char,
        num_epochs=20  # Train for 20 epochs
    )
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'training_history': training_history,
    }, 'hangman_bilstm_model.pth')
    
    print("Model saved as 'hangman_bilstm_model.pth'")


if __name__ == "__main__":
    main()