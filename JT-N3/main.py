from collections import Counter
import argparse
import os
import re


def create_profile(text, max_ngrams=300, min_n=1, max_n=5):
    # Tokenize
    tokens = re.findall(r"[a-zA-Z']+", text.lower())

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-gram based language detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command - create profile for a specific language
    train_parser = subparsers.add_parser('train', help='Create a language profile')
    train_parser.add_argument('language', help='Language name (e.g., english, slovenian)')
    train_parser.add_argument('--corpus', required=True, help='Path to corpus file')
    train_parser.add_argument('--output', required=True, help='Path to save the profile')

    # Classify command - detect language of a document
    classify_parser = subparsers.add_parser('classify', help='Classify a document')
    classify_parser.add_argument('--text', required=True, help='Path to text file to classify')
    classify_parser.add_argument('--profiles', required=True, help='Path to directory with profiles')

    args = parser.parse_args()

    if args.command == 'train':
        # Train a specific language profile
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        try:
            with open(args.corpus, 'r', encoding='utf-8') as f:
                text = f.read()

            profile = create_profile(text)

            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(profile))

            print(f"Created profile for {args.language} with {len(profile)} n-grams")
            print(f"Profile saved to {args.output}")

        except FileNotFoundError:
            print(f"Corpus file not found: {args.corpus}")

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
# https://www.corpusdata.org/formats.asp English&Spanish

