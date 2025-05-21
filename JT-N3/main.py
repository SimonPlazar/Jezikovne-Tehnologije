from collections import Counter
import argparse
import os
import re


def split_corpus_from_file(file_path, chunk_size=100000):
    with open(file_path, 'r', encoding='utf-8') as f:
        current_chunk = []
        current_size = 0

        for line in f:
            line_size = len(line)

            if current_size + line_size > chunk_size and current_chunk:
                yield ''.join(current_chunk)
                current_chunk = []
                current_size = 0

            # Add line to current chunk
            current_chunk.append(line)
            current_size += line_size

        # Yield the last chunk if there is one
        if current_chunk:
            yield ''.join(current_chunk)


def train_with_large_corpus(corpus_file, max_ngrams=300, min_n=1, max_n=5, chunk_size=100000):
    # Initialize a counter for all n-grams
    all_ngram_counts = Counter()

    # Process the corpus in chunks
    for chunk in split_corpus_from_file(corpus_file, chunk_size):
        # For each chunk, extract tokens and generate n-grams
        # tokens = re.findall(r"[a-zA-Z']+", chunk.lower())
        tokens = re.findall(r"[^\W\d_]+", chunk.lower())

        for token in tokens:
            for n in range(min_n, max_n + 1):
                # Handle unigrams specially
                if n == 1:
                    for char in token:
                        all_ngram_counts[char] += 1
                    continue

                # Normal n-gram processing with padding
                padded_token = '_' + token + '_' * (n - 1)
                for i in range(len(padded_token) - n + 1):
                    ngram = padded_token[i:i + n]
                    all_ngram_counts[ngram] += 1

    # Get the final profile
    most_common = all_ngram_counts.most_common(max_ngrams)
    profile = [ngram for ngram, _ in most_common]

    return profile


def create_profile(text, max_ngrams=300, min_n=1, max_n=5):
    # Tokenize
    # tokens = re.findall(r"[a-zA-Z']+", text.lower())
    tokens = re.findall(r"[^\W\d_]+", text.lower()) # Only keep alphabetic characters

    # Generate n-grams from tokens with proper padding
    all_ngrams = []
    for token in tokens:
        # Generate n-grams of all required lengths
        for n in range(min_n, max_n + 1):
            # For unigrams, use characters without padding
            if n == 1:
                for char in token:
                    all_ngrams.append(char)
                continue

            # For n > 1, use padding as specified
            padded_token = '_' + token + '_' * (n - 1)

            # Generate all n-grams for this padded token
            for i in range(len(padded_token) - n + 1):
                ngram = padded_token[i:i + n]
                all_ngrams.append(ngram)

    # Count n-gram frequencies
    ngram_counts = Counter(all_ngrams)

    # Sort by frequency and keep top max_ngrams
    most_common = ngram_counts.most_common(max_ngrams)
    profile = [ngram for ngram, _ in most_common]

    return profile


def calculate_profile_distance(doc_profile, category_profile):
    distance = 0
    # Maximum penalty for n-grams not found in category profile
    max_penalty = len(category_profile)

    # Create a dictionary to quickly look up the rank of each n-gram in the category profile
    category_ranks = {ngram: rank for rank, ngram in enumerate(category_profile)}

    # Calculate out-of-place measure for each n-gram in the document profile
    for doc_rank, ngram in enumerate(doc_profile):
        if ngram in category_ranks:
            # Calculate how far out of place this n-gram is
            cat_rank = category_ranks[ngram]
            out_of_place = abs(doc_rank - cat_rank)
            distance += out_of_place
        else:
            # N-gram not found in category profile, apply maximum penalty
            distance += max_penalty

    return distance


def classify_document(text, profiles):
    # Generate profile for the document
    doc_profile = create_profile(text)

    # Calculate distance to each language profile
    distances = {}
    for language, lang_profile in profiles.items():
        distance = calculate_profile_distance(doc_profile, lang_profile)
        distances[language] = distance

    # Find the language with the minimum distance
    detected_lang = min(distances, key=distances.get)

    return detected_lang, distances


def classify_test_files(profiles_dir='Profiles', test_files_dir='test_files'):
    # Load all profiles
    profiles = {}
    for filename in os.listdir(profiles_dir):
        if filename.endswith('_profile.txt'):
            lang = filename.replace('_profile.txt', '')
            profile_path = os.path.join(profiles_dir, filename)
            with open(profile_path, 'r', encoding='utf-8') as f:
                profiles[lang] = f.read().splitlines()

    if not profiles:
        print(f"No profiles found in {profiles_dir}")
        return

    if not os.path.exists(test_files_dir):
        print(f"Test files directory {test_files_dir} does not exist.")
        return

    # Process each file in the test directory
    for filename in os.listdir(test_files_dir):
        file_path = os.path.join(test_files_dir, filename)
        if os.path.isfile(file_path):
            try:
                # Read the document
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_text = f.read()

                # Classify the document
                detected_lang, _ = classify_document(doc_text, profiles)

                # Output result
                print(f"{filename} - {detected_lang}")

            except Exception as e:
                print(f"{filename} - Error: {e}")


if __name__ == "__main__":
    # classify_test_files()
    # exit(0)

    parser = argparse.ArgumentParser(description='N-gram based language detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command - create profile for a specific language
    train_parser = subparsers.add_parser('train', help='Create a language profile')
    train_parser.add_argument('language', help='Language name (e.g., english, slovenian...)')
    train_parser.add_argument('--corpus', required=True, help='Path to corpus file')
    train_parser.add_argument('--output', required=True, help='Path to directory to save profiles')
    train_parser.add_argument('--max_ngrams', type=int, default=300, help='Maximum number of n-grams to keep')

    # Classify command - detect language of a document
    classify_parser = subparsers.add_parser('classify', help='Classify a document')
    classify_parser.add_argument('--text', required=True, help='Path to text file to classify')
    classify_parser.add_argument('--profiles', required=True, help='Path to directory with saved profiles')

    args = parser.parse_args()

    if args.command == 'train':
        # Train a specific language profile
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        try:
            # with open(args.corpus, 'r', encoding='utf-8') as f:
            #     text = f.read()
            # profile = create_profile(text)

            profile = train_with_large_corpus(args.corpus, args.max_ngrams)

            print(f"Created profile for {args.language} with {len(profile)} n-grams")

            profile_path = os.path.join(args.output, f"{args.language}_profile.txt")
            with open(profile_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(profile))

            print(f"Profile saved to {args.output}")

        except FileNotFoundError:
            print(f"File not found")

    elif args.command == 'classify':
        # Classification mode
        profiles = {}

        try:
            # Load all profiles from the directory
            for filename in os.listdir(args.profiles):
                if filename.endswith('_profile.txt'):
                    lang = filename.replace('_profile.txt', '')
                    profile_path = os.path.join(args.profiles, filename)
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        profiles[lang] = f.read().splitlines()

            if not profiles:
                print(f"No profiles found in {args.profiles}")
                exit(1)

            # Read the document to classify
            with open(args.text, 'r', encoding='utf-8') as f:
                doc_text = f.read()

            detected_lang, distances = classify_document(doc_text, profiles)

            print(f"Document language: {detected_lang}")
            print("Distances from each language profile:")
            for lang, dist in sorted(distances.items(), key=lambda x: x[1]):
                print(f"  {lang}: {dist}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
    else:
        parser.print_help()

# corpus
# https://wortschatz.uni-leipzig.de/en/download/German
