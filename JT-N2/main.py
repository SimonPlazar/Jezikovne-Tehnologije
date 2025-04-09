import math
import os
import pickle
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

class NGramModel:
    """Base class for n-gram language models"""

    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.vocabulary = set()

    def tokenize(self, text):
        """Tokenize text using NLTK"""
        return [token.lower() for token in word_tokenize(text)]

    def train_from_files(self, folder_path):
        """Incrementally train model from multiple files"""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' not found")

        file_count = 0
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"Processing {filename}...")
                    file_content = f.read()
                    self.train(file_content)
                    file_count += 1

        if file_count == 0:
            raise ValueError(f"No files found in '{folder_path}' folder")

        print(f"Successfully processed {file_count} files.")
        return file_count

    def train(self, corpus):
        """Train model on text corpus"""
        tokens = self.tokenize(corpus)
        self.vocabulary.update(tokens)
        self.vocabulary.update(["<s>", "</s>"])

        # Count n-grams for all orders up to n
        for i in range(1, self.n + 1):
            padded = ["<s>"] * (i - 1) + tokens + ["</s>"]
            text_ngrams = list(ngrams(padded, i))
            self.ngram_counts[i].update(text_ngrams)

        self.finalize_training()

    def finalize_training(self):
        """Hook for derived classes to perform post-training calculations"""
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def perplexity(self, corpus):
        tokens = self.tokenize(corpus)
        log_prob_sum = 0
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        text_ngrams = list(ngrams(padded, self.n))

        for ngram in text_ngrams:
            context, token = ngram[:-1], ngram[-1]
            prob = self.probability(token, context)
            if prob > 0:
                log_prob_sum += math.log2(prob)

        return 2 ** (-log_prob_sum / len(text_ngrams))

    def probability(self, token, context):
        raise NotImplementedError


class GoodTuringModel(NGramModel):
    def __init__(self, n, max_count=5):
        super().__init__(n)
        self.max_count = max_count
        self.count_of_counts = defaultdict(int)

    def finalize_training(self):
        """Calculate count-of-counts for Good-Turing estimation"""
        self.count_of_counts.clear()
        for count in self.ngram_counts[self.n].values():
            if count <= self.max_count + 1:
                self.count_of_counts[count] += 1

    def good_turing_estimate(self, count):
        """Apply Good-Turing smoothing to a count"""
        if count == 0:
            # Zero count smoothing
            return self.count_of_counts[1] / sum(self.ngram_counts[self.n].values())

        if count > self.max_count or count + 1 not in self.count_of_counts:
            return count / sum(self.ngram_counts[self.n].values())

        # Good-Turing formula: c* = (c+1) * N_{c+1} / N_c
        return (count + 1) * self.count_of_counts[count + 1] / self.count_of_counts[count]

    def probability(self, token, context):
        ngram = context + (token,)
        count = self.ngram_counts[self.n].get(ngram, 0)
        context_count = self.ngram_counts[self.n - 1].get(context, 0)

        if context_count == 0:
            return 1 / len(self.vocabulary)

        gt_estimate = self.good_turing_estimate(count)
        return gt_estimate / context_count


class KneserNeyModel(NGramModel):
    def __init__(self, n, discount=0.75):
        super().__init__(n)
        self.discount = discount
        self.continuation_counts = defaultdict(int)
        self.context_types = defaultdict(set)
        self.following_words = defaultdict(set)

    def finalize_training(self):
        """Calculate continuation counts for Kneser-Ney smoothing"""
        self.context_types.clear()
        self.continuation_counts.clear()
        self.following_words.clear()

        for ngram in self.ngram_counts[self.n]:
            if self.n > 1:
                context = ngram[:-1]
                word = ngram[-1]
                self.context_types[word].add(context)
                self.following_words[context].add(word)

        for word in self.context_types:
            self.continuation_counts[word] = len(self.context_types[word])

    def probability(self, token, context):
        if self.n == 1:
            # Unigram case
            count = self.ngram_counts[1].get((token,), 0)
            total = sum(self.ngram_counts[1].values())
            return max(count - self.discount, 0) / total + \
                (self.discount / total * len(self.vocabulary))

        # Higher order n-grams
        ngram = context + (token,)
        count = self.ngram_counts[self.n].get(ngram, 0)
        context_count = self.ngram_counts[self.n - 1].get(context, 0)

        if context_count == 0:
            # Backoff to lower order
            if len(context) > 1:
                return self.probability(token, context[1:])
            else:
                return self.probability(token, ())

        # Count of unique words following this context
        following_types = len(self.following_words[context])

        # Calculate lambda (normalization factor)
        lambda_factor = (self.discount * following_types) / context_count

        # Get lower-order probability (recursively)
        if len(context) > 1:
            lower_prob = self.probability(token, context[1:])
        else:
            continuation_sum = sum(self.continuation_counts.values())
            lower_prob = self.continuation_counts[token] / continuation_sum if continuation_sum else 0

        return max(count - self.discount, 0) / context_count + lambda_factor * lower_prob


def main():
    corpus_folder = "korpus"
    if not os.path.exists(corpus_folder):
        print(f"Error: Folder '{corpus_folder}' not found!")
        return

    print("Training models...")

    # Initialize models
    models = {
        "Bigram GT": GoodTuringModel(2),
        "Trigram GT": GoodTuringModel(3),
        "Bigram KN": KneserNeyModel(2),
        "Trigram KN": KneserNeyModel(3)
    }

    # Get sample text for evaluation
    sample_text = ""
    sample_file = os.path.join(corpus_folder, os.listdir(corpus_folder)[0])
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_text = f.read()

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.train_from_files(corpus_folder)
        model.save(f"{name.lower().replace(' ', '_')}.model")
        print(f"{name} perplexity: {model.perplexity(sample_text):.2f}")

    # Test loading
    print("\nTesting model loading...")
    loaded_model = KneserNeyModel.load("trigram_kn.model")
    print(f"Loaded model perplexity: {loaded_model.perplexity(sample_text):.2f}")


if __name__ == "__main__":
    main()