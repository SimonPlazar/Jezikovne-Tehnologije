import math
import os
import pickle
import random
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re


def tokenize_text(text):
    # return [token.lower() for token in word_tokenize(text)]
    return [
        token.lower()
        for token in word_tokenize(text)
        if re.fullmatch(r"[^\W\d_]+", token.lower())
    ]


def get_ngrams_from_tokens(tokens, n):
    return list(ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))


def process_corpus(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return []

    all_tokens = []
    file_count = 0

    for filename in os.listdir(folder_path)[:40]:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"Processing {filename}...")
                    file_content = f.read()
                    tokens = tokenize_text(file_content)
                    all_tokens.extend(tokens)
                    file_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Successfully processed {file_count} files.")
    return all_tokens


def build_ngram_counts(tokens, max_n):
    ngram_counts = defaultdict(Counter)
    vocabulary = set(tokens + ["<s>", "</s>"])

    # Count n-grams
    for i in range(1, max_n + 1):
        ngrams_list = get_ngrams_from_tokens(tokens, i)
        ngram_counts[i].update(ngrams_list)

    return ngram_counts, vocabulary


def calculate_perplexity(model, corpus, n):
    tokens = tokenize_text(corpus)
    log_prob_sum = 0
    ngrams_list = get_ngrams_from_tokens(tokens, n)

    if not ngrams_list:
        return float('inf')

    for ngram in ngrams_list:
        context, token = ngram[:-1], ngram[-1]
        prob = model.probability(token, context)
        if prob > 0:
            log_prob_sum += math.log2(prob)
        else:
            log_prob_sum += math.log2(1e-10)

    # 2^(-average log probability)
    return 2 ** (-log_prob_sum / len(ngrams_list))


class NGramModel:
    def __init__(self, ngram_counts, vocabulary, n):
        self.ngram_counts = ngram_counts
        self.vocabulary = vocabulary
        self.n = n

    def probability(self, token, context):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class GoodTuringModel(NGramModel):
    def __init__(self, ngram_counts, vocabulary, n, max_count=5):
        super().__init__(ngram_counts, vocabulary, n)
        self.max_count = max_count
        self.count_of_counts = self._calculate_count_of_counts()
        self.total_ngrams = sum(ngram_counts[n].values())

        self.vocab_size = len(vocabulary) - 2  # remove <s>, </s>
        self.possible_ngrams = self.vocab_size ** n
        self.Nb = len(ngram_counts[n])
        self.N0 = max(self.possible_ngrams - self.Nb, 1)

        self.adjusted_counts = {}
        for c in range(0, max_count + 2):
            Nc = self.count_of_counts.get(c, 0)
            Nc1 = self.count_of_counts.get(c + 1, 0)
            if Nc > 0:
                self.adjusted_counts[c] = (c + 1) * Nc1 / Nc

    def _calculate_count_of_counts(self):
        count_of_counts = defaultdict(int)
        for count in self.ngram_counts[self.n].values():
            if count <= self.max_count + 1:
                count_of_counts[count] += 1
        return count_of_counts

    def _good_turing_estimate(self, count):
        if count in self.adjusted_counts:
            return self.adjusted_counts[count] / self.total_ngrams
        elif count == 0:
            return self.adjusted_counts.get(0, 1e-10) / self.N0
        else:
            return count / self.total_ngrams

    def probability(self, token, context):
        ngram = context + (token,)
        count = self.ngram_counts[self.n].get(ngram, 0)
        return self._good_turing_estimate(count)


class KneserNeyModel(NGramModel):
    def __init__(self, ngram_counts, vocabulary, n, discount=0.75):
        super().__init__(ngram_counts, vocabulary, n)
        self.discount = discount
        self._calculate_continuation_stats()

    # For each word tracks how many different contexts it appears in
    def _calculate_continuation_stats(self):
        self.context_types = defaultdict(set)
        self.following_words = defaultdict(set)

        for ngram in self.ngram_counts[self.n]:
            if self.n > 1:
                context = ngram[:-1]
                word = ngram[-1]
                self.context_types[word].add(context)
                self.following_words[context].add(word)

        self.continuation_counts = {word: len(contexts) for word, contexts in self.context_types.items()}

    def probability(self, token, context):
        # Unigram case (no context or explicitly unigram model)
        if len(context) == 0 or self.n == 1:
            count = self.ngram_counts[1].get((token,), 0)
            total = sum(self.ngram_counts[1].values())
            if total == 0:
                return 1 / len(self.vocabulary) if self.vocabulary else 1e-10
            return max(count - self.discount, 0) / total + (self.discount * len(self.vocabulary) / total)

        # Get counts for this n-gram and its context
        ngram = context + (token,)
        count = self.ngram_counts[self.n].get(ngram, 0)
        context_count = self.ngram_counts[self.n - 1].get(context, 0)

        # Back off to lower order if context not seen
        if context_count == 0:
            return self.probability(token, context[1:] if len(context) > 1 else ())

        # P(w|context) = max(count(context,w) - d, 0)/count(context) + λ × P_lower(w)
        discount_term = max(count - self.discount, 0) / context_count
        following_count = len(self.following_words.get(context, set()))
        lambda_factor = (self.discount * following_count) / context_count
        lower_prob = self.probability(token, context[1:] if len(context) > 1 else ())

        return discount_term + (lambda_factor * lower_prob)


def predict_next_token(model, text, top_k=5):
    tokens = tokenize_text(text)

    # Get the appropriate context length based on model's n-gram size
    context_size = model.n - 1
    if len(tokens) >= context_size:
        context = tuple(tokens[-context_size:])
    else:
        # Pad with start tokens if needed
        context = tuple(["<s>"] * (context_size - len(tokens)) + tokens)

    # Calculate probability for each word in vocabulary
    word_probs = {}
    for word in model.vocabulary:
        if word not in ["<s>", "</s>"]:  # Skip start/end tokens
            word_probs[word] = model.probability(word, context)

    # Return top k predictions
    total_prob = sum(word_probs.values())
    print("Total probability:", total_prob)
    if total_prob > 0:
        word_probs = {w: p / total_prob for w, p in word_probs.items()}
    top_predictions = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top_predictions


if __name__ == "__main__":
    corpus_folder = "korpus"
    if not os.path.exists(corpus_folder):
        print(f"Error: Folder '{corpus_folder}' not found!")
        exit(1)

    # Process corpus once
    print("Processing corpus...")
    all_tokens = process_corpus(corpus_folder)
    if not all_tokens:
        print("No content found in corpus!")
        exit(1)

    # Build n-gram counts up to trigrams
    max_n = 3
    ngram_counts, vocabulary = build_ngram_counts(all_tokens, max_n)

    # Create models
    print("\nCreating models...")
    models = {
        # "Bigram GT": GoodTuringModel(ngram_counts, vocabulary, 2),
        "Trigram GT": GoodTuringModel(ngram_counts, vocabulary, 3),
        # "Bigram KN": KneserNeyModel(ngram_counts, vocabulary, 2),
        # "Trigram KN": KneserNeyModel(ngram_counts, vocabulary, 3)
    }

    sample_text = ""
    # Get sample text for perplexity
    try:
        sample_files = os.listdir(corpus_folder)
        if not sample_files:
            print("No files found in corpus folder!")
            exit(1)

        sample_file = os.path.join(corpus_folder, random.choice(sample_files))
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_text = f.read()
    except Exception as e:
        print(f"Error reading sample file: {e}")
        sample_text = ""

    # Save models and calculate perplexity
    for name, model in models.items():
        model.save(f"{name.lower().replace(' ', '_')}.model")

        if sample_text:
            perplexity = calculate_perplexity(model, sample_text, model.n)
            print(f"{name} perplexity: {perplexity:.2f}")

    # Test loading
    if os.path.exists("trigram_gt.model"):
        print("\nTesting model loading...")
        loaded_model = KneserNeyModel.load("trigram_gt.model")
        if sample_text:
            perplexity = calculate_perplexity(loaded_model, sample_text, loaded_model.n)
            print(f"Loaded model perplexity: {perplexity:.2f}")

    print("\nNext token prediction demo:")
    # selected_model = models["Trigram KN"]
    selected_model = loaded_model
    text = ""
    while True:
        user_input = input("\nEnter text (or 'q' to quit | 'r' to reset ): ")
        if user_input.lower() == "q":
            break

        if user_input.lower() == "r":
            text = ""
            continue

        if not text:
            text = user_input
        else:
            text += " " + user_input
        print(f"\nInput text: '{text}'")
        tokens = text.split()
        context = tuple(tokens[-(selected_model.n - 1) - 1:-1]) if len(tokens) >= selected_model.n else tuple(
            tokens[:-1])
        token = tokens[-1] if tokens else ""
        print(f"Probability for {list(context) + [token]}: {selected_model.probability(token, context)}")
        predictions = predict_next_token(selected_model, text)
        print(f"Top 5 predicted next words:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"{i}. '{word}' (probability: {prob:.6f})")
