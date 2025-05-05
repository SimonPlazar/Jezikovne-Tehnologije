import os
import re
import argparse
from collections import Counter


def generate_ngrams(token, n_range=(1, 5)):
    """Generate all n-grams of specified lengths from a token."""
    # Pad the token with spaces
    padded_token = ' ' + token + ' '
    ngrams = []

    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(padded_token) - n + 1):
            ngrams.append(padded_token[i:i + n])

    return ngrams


def create_profile(text, max_ngrams=300):
    """Create an n-gram frequency profile from text."""
    # Tokenize the text (keeping only letters and apostrophes)
    tokens = re.findall(r"[a-zA-Z']+", text)

    # Generate all n-grams from the tokens
    all_ngrams = []
    for token in tokens:
        all_ngrams.extend(generate_ngrams(token.lower()))

    # Count n-gram frequencies
    ngram_counts = Counter(all_ngrams)

    # Sort by frequency and keep the top max_ngrams
    profile = [ngram for ngram, _ in ngram_counts.most_common(max_ngrams)]

    return profile


def calculate_distance(doc_profile, lang_profile, max_penalty=len):
    """Calculate the out-of-place distance between two profiles."""
    distance = 0
    max_rank = len(lang_profile)

    for i, ngram in enumerate(doc_profile):
        if ngram in lang_profile:
            # Calculate how far out of place the n-gram is
            lang_rank = lang_profile.index(ngram)
            distance += abs(i - lang_rank)
        else:
            # Apply maximum penalty for n-grams not in the language profile
            distance += max_rank

    return distance


def train_languages(training_dir, languages, max_ngrams=300):
    """Build profiles for specified languages."""
    profiles = {}

    for lang in languages:
        filename = os.path.join(training_dir, f"{lang}.txt")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            profiles[lang] = create_profile(text, max_ngrams)
            print(f"Created profile for {lang} with {len(profiles[lang])} n-grams")
        except FileNotFoundError:
            print(f"Training file for {lang} not found: {filename}")

    return profiles


def classify_document(doc_text, language_profiles):
    """Classify a document based on language profiles."""
    doc_profile = create_profile(doc_text)

    distances = {}
    for lang, profile in language_profiles.items():
        distances[lang] = calculate_distance(doc_profile, profile)

    # Find the language with the minimum distance
    detected_lang = min(distances, key=distances.get)
    return detected_lang, distances


def main():
    parser = argparse.ArgumentParser(description='N-gram based language detection')
    parser.add_argument('--train', action='store_true', help='Train language profiles')
    parser.add_argument('--classify', help='Classify a document file')
    parser.add_argument('--training-dir', default='training', help='Directory with training files')
    parser.add_argument('--profiles-dir', default='profiles', help='Directory to save profiles')

    args = parser.parse_args()

    languages = ['english', 'slovenian', 'german', 'spanish', 'croatian']

    if args.train:
        # Training mode - build profiles for all languages
        os.makedirs(args.profiles_dir, exist_ok=True)
        profiles = train_languages(args.training_dir, languages)

        # Save profiles (simplified approach - in practice use pickle or json)
        for lang, profile in profiles.items():
            with open(os.path.join(args.profiles_dir, f"{lang}_profile.txt"), 'w', encoding='utf-8') as f:
                f.write('\n'.join(profile))

        print("Training completed. Profiles saved.")

    elif args.classify:
        # Classification mode
        # Load profiles
        profiles = {}
        for lang in languages:
            profile_path = os.path.join(args.profiles_dir, f"{lang}_profile.txt")
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profiles[lang] = f.read().splitlines()
            except FileNotFoundError:
                print(f"Profile for {lang} not found. Please train first.")
                return

        # Classify the document
        try:
            with open(args.classify, 'r', encoding='utf-8') as f:
                doc_text = f.read()

            detected_lang, distances = classify_document(doc_text, profiles)

            print(f"Document language: {detected_lang}")
            print("Distances from each language profile:")
            for lang, dist in sorted(distances.items(), key=lambda x: x[1]):
                print(f"  {lang}: {dist}")

        except FileNotFoundError:
            print(f"Document file not found: {args.classify}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()