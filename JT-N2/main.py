import math
import os
import pickle
import re
from collections import Counter, defaultdict


def tokenize(text):
    """Tokenize text into words."""
    return re.findall(r'\w+', text.lower())


def get_ngrams(tokens, n):
    """Generate n-grams from tokens with start/end markers."""
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    return [tuple(padded[i:i + n]) for i in range(len(padded) - n + 1)]


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.vocabulary = set()

    def train(self, corpus):
        tokens = tokenize(corpus)
        self.vocabulary = set(tokens) | {"<s>", "</s>"}

        # Count n-grams for all orders up to n
        for i in range(1, self.n + 1):
            ngrams = get_ngrams(tokens, i)
            for ngram in ngrams:
                self.ngram_counts[i][ngram] += 1

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def perplexity(self, corpus):
        tokens = tokenize(corpus)
        log_prob_sum = 0
        ngrams = get_ngrams(tokens, self.n)

        for ngram in ngrams:
            context, token = ngram[:-1], ngram[-1]
            prob = self.probability(token, context)
            if prob > 0:
                log_prob_sum += math.log2(prob)

        return 2 ** (-log_prob_sum / len(ngrams))

    def probability(self, token, context):
        raise NotImplementedError


class GoodTuringModel(NGramModel):
    def __init__(self, n, max_count=5):
        super().__init__(n)
        self.max_count = max_count
        self.count_of_counts = defaultdict(int)

    def train(self, corpus):
        super().train(corpus)

        # Calculate count-of-counts
        for count in self.ngram_counts[self.n].values():
            if count <= self.max_count + 1:
                self.count_of_counts[count] += 1

    def good_turing_estimate(self, count):
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
        # For Kneser-Ney continuation counting
        self.continuation_counts = defaultdict(int)
        self.context_types = defaultdict(set)

    def train(self, corpus):
        super().train(corpus)

        # Calculate continuation counts for KN
        for ngram in self.ngram_counts[self.n]:
            if self.n > 1:
                context = ngram[:-1]
                word = ngram[-1]
                self.context_types[word].add(context)

        # Count number of different contexts word appears in
        for word in self.context_types:
            self.continuation_counts[word] = len(self.context_types[word])

    def probability(self, token, context):
        if self.n == 1:
            # Unigram case - use MLE with smoothing
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
        following_types = len([1 for ng in self.ngram_counts[self.n]
                               if ng[:-1] == context])

        # Calculate lambda (normalization factor)
        lambda_factor = (self.discount * following_types) / context_count

        # Get lower-order probability (recursively)
        lower_prob = self.probability(token, context[1:]) if len(context) > 1 else \
            self.continuation_counts[token] / sum(self.continuation_counts.values())

        return max(count - self.discount, 0) / context_count + lambda_factor * lower_prob


if __name__ == '__main__':
    # Load training corpus from multiple files in the korpus folder
    corpus = ""
    korpus_folder = "korpus"

    if not os.path.exists(korpus_folder):
        print(f"Error: Folder '{korpus_folder}' not found!")
        exit(1)

    file_count = 0
    for filename in os.listdir(korpus_folder):
        file_path = os.path.join(korpus_folder, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    corpus += file_content + " "  # Add space between files
                    file_count += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    if file_count == 0:
        print(f"No files found in '{korpus_folder}' folder!")
        exit(1)

    print(f"Successfully loaded {file_count} files from '{korpus_folder}' folder.")

    # Train models
    print("Training models...")
    # Good-Turing models
    bigram_gt = GoodTuringModel(2)
    bigram_gt.train(corpus)
    bigram_gt.save('bigram_gt.model')

    trigram_gt = GoodTuringModel(3)
    trigram_gt.train(corpus)
    trigram_gt.save('trigram_gt.model')

    # Kneser-Ney models
    bigram_kn = KneserNeyModel(2)
    bigram_kn.train(corpus)
    bigram_kn.save('bigram_kn.model')

    trigram_kn = KneserNeyModel(3)
    trigram_kn.train(corpus)
    trigram_kn.save('trigram_kn.model')

    # Calculate perplexity
    print("\nPerplexity on training corpus:")
    print(f"Bigram Good-Turing: {bigram_gt.perplexity(corpus):.2f}")
    print(f"Trigram Good-Turing: {trigram_gt.perplexity(corpus):.2f}")
    print(f"Bigram Kneser-Ney: {bigram_kn.perplexity(corpus):.2f}")
    print(f"Trigram Kneser-Ney: {trigram_kn.perplexity(corpus):.2f}")

    # Demonstrate loading from file
    print("\nLoading model from file and calculating perplexity...")
    loaded_model = KneserNeyModel.load('trigram_kn.model')
    print(f"Loaded Trigram KN perplexity: {loaded_model.perplexity(corpus):.2f}")
