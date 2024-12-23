# import nltk
# from nltk.corpus import brown
# from collections import Counter, defaultdict
# import numpy as np
# import re
#
# from sklearn.metrics import confusion_matrix
#
# START_TAG = '<START>'
# END_TAG = '<STOP>'
# DEFAULT_TAG = 'NN'  # Most likely tag for unknown words (NN)
#
#
# def download_and_load_corpus():
#     """
#         Downloads the Brown corpus and loads it with the universal tagset.
#         Returns the tagged sentences from the 'news' category.
#         """
#     nltk.download('brown')
#     return brown.tagged_sents(categories='news')
#
#
# def split_corpus(tagged_sentences):
#     """
#         Splits the corpus into training and testing sets (90% train, 10% test).
#
#         Parameters:
#             tagged_sentences (list): List of tagged sentences.
#
#         Returns:
#             tuple: Training and testing data.
#         """
#     split_point = int(len(tagged_sentences) * 0.9)
#     return tagged_sentences[:split_point], tagged_sentences[split_point:]
#
#
# def compute_most_likely_tags(train_data):
#     """
#         Computes the most likely tag for each word in the training data.
#
#         Parameters:
#             train_data (list): Training data with tagged sentences.
#
#         Returns:
#             dict: Mapping of words to their most likely tags.
#         """
#     word_tag_counts = defaultdict(Counter)
#     for sentence in train_data:
#         for word, tag in sentence:
#             word_tag_counts[word][tag] += 1
#     return {word: tag_counts.most_common(1)[0][0] for word, tag_counts in word_tag_counts.items()}
#
#
# def evaluate_baseline(test_data, most_likely_tag):
#     """
#         Evaluates the baseline tagging model using the most likely tags.
#
#         Parameters:
#             test_data (list): Testing data with tagged sentences.
#             most_likely_tag (dict): Mapping of words to their most likely tags.
#
#         Returns:
#             dict: Error rates for total, known, and unknown words.
#         """
#     total, correct, known_errors, unknown_errors = 0, 0, 0, 0
#     unknown_words, known_words = 0, 0
#     for sentence in test_data:
#         for word, true_tag in sentence:
#             total += 1
#             if word in most_likely_tag:
#                 known_words += 1
#                 if most_likely_tag[word] == true_tag:
#                     correct += 1
#                 else:
#                     known_errors += 1
#             else:
#                 unknown_words += 1
#                 if DEFAULT_TAG == true_tag:
#                     correct += 1
#                 else:
#                     unknown_errors += 1
#
#     return {
#         "total_error_rate": 1 - correct / total,
#         "known_error_rate": known_errors / known_words if known_words > 0 else 0,
#         "unknown_error_rate": unknown_errors / unknown_words if unknown_words > 0 else 0
#     }
#
#
# def compute_hmm_probabilities(train_data):
#     """
#         Computes transition and emission probabilities for HMM.
#
#         Parameters:
#             train_data (list): Training data with tagged sentences.
#
#         Returns:
#             tuple: Transition probabilities, emission probabilities, and set of tags.
#         """
#     tag_bigram_counts = Counter()
#     tag_counts = Counter()
#     emission_counts = defaultdict(Counter)
#
#     for sentence in train_data:
#         previous_tag = START_TAG
#         for word, tag in sentence:
#             tag_bigram_counts[(previous_tag, tag)] += 1
#             tag_counts[previous_tag] += 1
#             emission_counts[tag][word] += 1
#             previous_tag = tag
#         tag_counts[previous_tag] += 1
#         tag_bigram_counts[(previous_tag, END_TAG)] += 1
#
#     transition_probs = {
#         (prev, curr): count / tag_counts[prev]
#         for (prev, curr), count in tag_bigram_counts.items()
#     }
#
#     emission_probs = {
#         tag: {word: count / sum(word_counts.values()) for word, count in word_counts.items()}
#         for tag, word_counts in emission_counts.items()
#     }
#
#     return transition_probs, emission_probs, set(tag_counts.keys())
#
#
# def add_one_smoothing(train_data):
#     """
#         Applies add-one smoothing to transition and emission probabilities.
#
#         Parameters:
#             train_data (list): Training data with tagged sentences.
#
#         Returns:
#             tuple: Smoothed transition probabilities, emission probabilities, and set of tags.
#         """
#     tag_bigram_counts = Counter()
#     tag_counts = Counter()
#     emission_counts = defaultdict(Counter)
#     vocabulary = set()
#
#     for sentence in train_data:
#         previous_tag = START_TAG
#         for word, tag in sentence:
#             vocabulary.add(word)
#             tag_bigram_counts[(previous_tag, tag)] += 1
#             tag_counts[previous_tag] += 1
#             emission_counts[tag][word] += 1
#             previous_tag = tag
#         tag_counts[previous_tag] += 1
#         tag_bigram_counts[(previous_tag, END_TAG)] += 1
#
#     vocabulary_size = len(vocabulary)
#
#     transition_probs = {
#         (prev, curr): (count + 1) / (tag_counts[prev] + len(tag_counts))
#         for (prev, curr), count in tag_bigram_counts.items()
#     }
#
#     emission_probs = {
#         tag: {word: (count + 1) / (sum(word_counts.values()) + vocabulary_size)
#               for word, count in word_counts.items()}
#         for tag, word_counts in emission_counts.items()
#     }
#
#     return transition_probs, emission_probs, set(tag_counts.keys())
#
#
# def viterbi(sentence, transition_probs, emission_probs, all_tags):
#     """
#         Implements the Viterbi algorithm for HMM-based tagging.
#
#         Parameters:
#             sentence (list): List of words in the sentence.
#             transition_probs (dict): Transition probabilities.
#             emission_probs (dict): Emission probabilities.
#             all_tags (set): Set of all possible tags.
#
#         Returns:
#             list: Sequence of predicted tags.
#         """
#     n = len(sentence)
#     m = len(all_tags)
#     index_to_tag = list(all_tags)
#     dp = np.zeros((n, m))
#     backpointers = np.zeros((n, m), dtype=int)
#
#     for i, tag in enumerate(all_tags):
#         dp[0][i] = transition_probs.get((START_TAG, tag), 1e-6) * emission_probs.get(tag, {}).get(sentence[0], 1e-6)
#
#     for t in range(1, n):
#         for j, tag in enumerate(all_tags):
#             max_prob, best_prev = max(
#                 (dp[t - 1][k] * transition_probs.get((prev_tag, tag), 1e-6) * emission_probs.get(tag, {}).get(
#                     sentence[t], 1e-6), k)
#                 for k, prev_tag in enumerate(all_tags)
#             )
#             dp[t][j] = max_prob
#             backpointers[t][j] = best_prev
#
#     best_last_tag = np.argmax(dp[-1])
#     tags = [index_to_tag[best_last_tag]]
#     for t in range(n - 1, 0, -1):
#         best_last_tag = backpointers[t][best_last_tag]
#         tags.append(index_to_tag[best_last_tag])
#
#     return tags[::-1]
#
#
# def evaluate_hmm(test_data, transition_probs, emission_probs, all_tags, most_likely_tag):
#     """
#         Evaluates the HMM tagging model.
#
#         Parameters:
#             test_data (list): Testing data with tagged sentences.
#             transition_probs (dict): Transition probabilities.
#             emission_probs (dict): Emission probabilities.
#             all_tags (set): Set of all possible tags.
#             most_likely_tag (dict): Mapping of words to their most likely tags.
#
#         Returns:
#             dict: Error rates for total, known, and unknown words.
#         """
#     total, correct, known_errors, unknown_errors = 0, 0, 0, 0
#     unknown_words, known_words = 0, 0
#
#     for sentence in test_data:
#         words, true_tags = zip(*sentence)
#         predicted_tags = viterbi(words, transition_probs, emission_probs, all_tags)
#         for word, true_tag, predicted_tag in zip(words, true_tags, predicted_tags):
#             total += 1
#             if word in most_likely_tag:
#                 known_words += 1
#                 if predicted_tag == true_tag:
#                     correct += 1
#                 else:
#                     known_errors += 1
#             else:
#                 unknown_words += 1
#                 if predicted_tag == true_tag:
#                     correct += 1
#                 else:
#                     unknown_errors += 1
#
#     return {
#         "total_error_rate": 1 - correct / total,
#         "known_error_rate": known_errors / known_words if known_words > 0 else 0,
#         "unknown_error_rate": unknown_errors / unknown_words if unknown_words > 0 else 0
#     }
#
#
# def compute_pseudo_words(word):
#     """
#        Maps words to pseudo-word categories based on their characteristics.
#
#        Parameters:
#            word (str): Input word.
#
#        Returns:
#            str: Pseudo-word category.
#        """
#     if re.fullmatch(r"\\d{2}", word):
#         return "<twoDigitNum>"  # Two digit number
#     elif re.fullmatch(r"\\d{4}", word):
#         return "<fourDigitNum>"  # Four digit year
#     elif re.search(r"[A-Z]\\d|\\d[A-Z]", word):
#         return "<containsDigitAndAlpha>"  # Contains letters and numbers
#     elif "-" in word and re.search(r"\\d", word):
#         return "<containsDigitAndDash>"  # Contains digits and dash
#     elif "/" in word and re.search(r"\\d", word):
#         return "<containsDigitAndSlash>"  # Contains digits and slash
#     elif "," in word and re.search(r"\\d", word):
#         return "<containsDigitAndComma>"  # Contains digits and comma
#     elif "." in word and re.search(r"\\d", word):
#         return "<containsDigitAndPeriod>"  # Contains digits and period
#     elif word.isdigit():
#         return "<othernum>"  # Other number
#     elif word.isupper():
#         return "<allCaps>"  # All uppercase
#     elif re.fullmatch(r"[A-Z]\\.", word):
#         return "<capPeriod>"  # Capitalized period (e.g., M.)
#     elif word[0].isupper():
#         return "<initCap>"  # Initial capitalized word
#     elif word.islower():
#         return "<lowercase>"  # Lowercase word
#     else:
#         return "<other>"  # Punctuation or other
#
#
# def generate_pseudo_word_data(train_data, test_data, threshold=5):
#     """
#         Replaces rare words with pseudo-words in the dataset.
#
#         Parameters:
#             train_data (list): Training data with tagged sentences.
#             test_data (list): Testing data with tagged sentences.
#             threshold (int): Frequency threshold for rare words.
#
#         Returns:
#             tuple: Training and testing data with pseudo-words.
#         """
#     word_counts = Counter(word for sentence in train_data for word, _ in sentence)
#     rare_words = {word for word, count in word_counts.items() if count < threshold}
#
#     def replace_with_pseudo_words(data):
#         new_data = []
#         for sentence in data:
#             new_sentence = []
#             for word, tag in sentence:
#                 if word in rare_words:
#                     new_word = compute_pseudo_words(word)
#                 else:
#                     new_word = word
#                 new_sentence.append((new_word, tag))
#             new_data.append(new_sentence)
#         return new_data
#
#     return replace_with_pseudo_words(train_data), replace_with_pseudo_words(test_data)
#
#
# def build_confusion_matrix(test_data, transition_probs, emission_probs, all_tags, most_likely_tag):
#     """
#        Constructs a confusion matrix for the HMM model.
#
#        Parameters:
#            test_data (list): Testing data with tagged sentences.
#            transition_probs (dict): Transition probabilities.
#            emission_probs (dict): Emission probabilities.
#            all_tags (set): Set of all possible tags.
#            most_likely_tag (dict): Mapping of words to their most likely tags.
#
#        Returns:
#            tuple: Confusion matrix and list of tags.
#        """
#     y_true = []
#     y_pred = []
#
#     for sentence in test_data:
#         words, true_tags = zip(*sentence)
#         predicted_tags = viterbi(words, transition_probs, emission_probs, all_tags)
#         y_true.extend(true_tags)
#         y_pred.extend(predicted_tags)
#
#     tag_list = sorted(list(all_tags))
#     cm = confusion_matrix(y_true, y_pred, labels=tag_list)
#     return cm, tag_list
#
#
# if __name__ == "__main__":
#     # Part (a) : Download and split test train
#     tagged_sentences = download_and_load_corpus()
#     train_data, test_data = split_corpus(tagged_sentences)
#
#     # Part (b): Baseline Evaluation
#     most_likely_tag = compute_most_likely_tags(train_data)
#     baseline_error_rate = evaluate_baseline(test_data, most_likely_tag)
#     print("Baseline Error Rate:", baseline_error_rate)
#
#     # Part (c): HMM Evaluation
#     transition_probs, emission_probs, all_tags = compute_hmm_probabilities(train_data)
#     hmm_error_rate = evaluate_hmm(test_data, transition_probs, emission_probs, all_tags, most_likely_tag)
#     print("HMM Error Rate:", hmm_error_rate)
#
#     # Part (d): Add-one Smoothing HMM Evaluation
#     smoothed_transition_probs, smoothed_emission_probs, smoothed_tags = add_one_smoothing(train_data)
#     smoothed_error_rate = evaluate_hmm(test_data, smoothed_transition_probs, smoothed_emission_probs, smoothed_tags,
#                                        most_likely_tag)
#     print("Smoothed HMM Error Rate:", smoothed_error_rate)
#
#     # Part (e) Pseudo-Words
#     pseudo_train_data, pseudo_test_data = generate_pseudo_word_data(train_data, test_data)
#
#     # Train using pseudo-words
#     pseudo_transition_probs, pseudo_emission_probs, pseudo_tags = compute_hmm_probabilities(pseudo_train_data)
#     pseudo_error_rate = evaluate_hmm(pseudo_test_data, pseudo_transition_probs, pseudo_emission_probs, pseudo_tags,
#                                      most_likely_tag)
#     print("Pseudo-Words HMM Error Rate:", pseudo_error_rate)
#
#     # Add-One Smoothing with Pseudo-Words
#     smoothed_pseudo_transition_probs, smoothed_pseudo_emission_probs, smoothed_pseudo_tags = add_one_smoothing(
#         pseudo_train_data)
#     smoothed_pseudo_error_rate = evaluate_hmm(pseudo_test_data, smoothed_pseudo_transition_probs,
#                                               smoothed_pseudo_emission_probs, smoothed_pseudo_tags, most_likely_tag)
#     print("Smoothed Pseudo-Words HMM Error Rate:", smoothed_pseudo_error_rate)
#
#     # Generate and plot confusion matrix
#     confusion_mat, tags = build_confusion_matrix(
#         pseudo_test_data, smoothed_pseudo_transition_probs, smoothed_pseudo_emission_probs, smoothed_pseudo_tags,
#         most_likely_tag
#     )
#     print("\nConfusion Matrix:")
#     print(confusion_mat)
#
#     # Error analysis
#     most_frequent_errors = [
#         (tags[i], tags[j], confusion_mat[i, j])
#         for i in range(len(tags))
#         for j in range(len(tags))
#         if i != j and confusion_mat[i, j] > 0
#     ]
#     most_frequent_errors.sort(key=lambda x: x[2], reverse=True)
#
#     print("\nMost Frequent Errors:")
#     for true_tag, predicted_tag, count in most_frequent_errors[:10]:
#         print(f"True Tag: {true_tag}, Predicted Tag: {predicted_tag}, Count: {count}")
import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict
import numpy as np
import re
from sklearn.metrics import confusion_matrix

START_TAG = '<START>'
END_TAG = '<STOP>'
DEFAULT_TAG = 'NN'  # Most likely tag for unknown words (NN)

def download_and_load_corpus():
    nltk.download('brown')
    return brown.tagged_sents(categories='news')

def split_corpus(tagged_sentences):
    split_point = int(len(tagged_sentences) * 0.9)
    return tagged_sentences[:split_point], tagged_sentences[split_point:]

def compute_most_likely_tags(train_data):
    word_tag_counts = defaultdict(Counter)
    for sentence in train_data:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
    return {word: tag_counts.most_common(1)[0][0] for word, tag_counts in word_tag_counts.items()}

def evaluate_baseline(test_data, most_likely_tag):
    total, correct, known_errors, unknown_errors = 0, 0, 0, 0
    unknown_words, known_words = 0, 0
    for sentence in test_data:
        for word, true_tag in sentence:
            total += 1
            if word in most_likely_tag:
                known_words += 1
                if most_likely_tag[word] == true_tag:
                    correct += 1
                else:
                    known_errors += 1
            else:
                unknown_words += 1
                if DEFAULT_TAG == true_tag:
                    correct += 1
                else:
                    unknown_errors += 1

    return {
        "total_error_rate": 1 - correct / total,
        "known_error_rate": known_errors / known_words if known_words > 0 else 0,
        "unknown_error_rate": unknown_errors / unknown_words if unknown_words > 0 else 0
    }

def compute_hmm_probabilities(train_data):
    tag_bigram_counts = Counter()
    tag_counts = Counter()
    emission_counts = defaultdict(Counter)

    for sentence in train_data:
        previous_tag = START_TAG
        for word, tag in sentence:
            tag_bigram_counts[(previous_tag, tag)] += 1
            tag_counts[previous_tag] += 1
            emission_counts[tag][word] += 1
            previous_tag = tag
        tag_counts[previous_tag] += 1
        tag_bigram_counts[(previous_tag, END_TAG)] += 1

    transition_probs = {
        (prev, curr): count / tag_counts[prev]
        for (prev, curr), count in tag_bigram_counts.items()
    }

    emission_probs = {
        tag: {word: count / sum(word_counts.values()) for word, count in word_counts.items()}
        for tag, word_counts in emission_counts.items()
    }

    return transition_probs, emission_probs, set(tag_counts.keys())

def evaluate_hmm(test_data, transition_probs, emission_probs, all_tags, most_likely_tag):
    """
        Evaluates the HMM tagging model.

        Parameters:
            test_data (list): Testing data with tagged sentences.
            transition_probs (dict): Transition probabilities.
            emission_probs (dict): Emission probabilities.
            all_tags (set): Set of all possible tags.
            most_likely_tag (dict): Mapping of words to their most likely tags.

        Returns:
            dict: Error rates for total, known, and unknown words.
        """
    total, correct, known_errors, unknown_errors = 0, 0, 0, 0
    unknown_words, known_words = 0, 0

    for sentence in test_data:
        words, true_tags = zip(*sentence)
        predicted_tags = viterbi(words, transition_probs, emission_probs, all_tags)
        for word, true_tag, predicted_tag in zip(words, true_tags, predicted_tags):
            total += 1
            if word in most_likely_tag:
                known_words += 1
                if predicted_tag == true_tag:
                    correct += 1
                else:
                    known_errors += 1
            else:
                unknown_words += 1
                if predicted_tag == true_tag:
                    correct += 1
                else:
                    unknown_errors += 1

    return {
        "total_error_rate": 1 - correct / total,
        "known_error_rate": known_errors / known_words if known_words > 0 else 0,
        "unknown_error_rate": unknown_errors / unknown_words if unknown_words > 0 else 0
    }

def add_one_smoothing(train_data):
    tag_bigram_counts = Counter()
    tag_counts = Counter()
    emission_counts = defaultdict(Counter)
    vocabulary = set()

    for sentence in train_data:
        previous_tag = START_TAG
        for word, tag in sentence:
            vocabulary.add(word)
            tag_bigram_counts[(previous_tag, tag)] += 1
            tag_counts[previous_tag] += 1
            emission_counts[tag][word] += 1
            previous_tag = tag
        tag_counts[previous_tag] += 1
        tag_bigram_counts[(previous_tag, END_TAG)] += 1

    vocabulary_size = len(vocabulary)

    transition_probs = {
        (prev, curr): (count + 1) / (tag_counts[prev] + len(tag_counts))
        for (prev, curr), count in tag_bigram_counts.items()
    }

    emission_probs = {
        tag: {word: (count + 1) / (sum(word_counts.values()) + vocabulary_size)
              for word, count in word_counts.items()}
        for tag, word_counts in emission_counts.items()
    }

    return transition_probs, emission_probs, set(tag_counts.keys())

def viterbi(sentence, transition_probs, emission_probs, all_tags):
    n = len(sentence)
    m = len(all_tags)
    index_to_tag = list(all_tags)
    dp = np.zeros((n, m))
    backpointers = np.zeros((n, m), dtype=int)

    for i, tag in enumerate(all_tags):
        dp[0][i] = transition_probs.get((START_TAG, tag), 1e-6) * emission_probs.get(tag, {}).get(sentence[0], 1e-6)

    for t in range(1, n):
        for j, tag in enumerate(all_tags):
            max_prob, best_prev = max(
                (dp[t - 1][k] * transition_probs.get((prev_tag, tag), 1e-6) * emission_probs.get(tag, {}).get(
                    sentence[t], 1e-6), k)
                for k, prev_tag in enumerate(all_tags)
            )
            dp[t][j] = max_prob
            backpointers[t][j] = best_prev

    best_last_tag = np.argmax(dp[-1])
    tags = [index_to_tag[best_last_tag]]
    for t in range(n - 1, 0, -1):
        best_last_tag = backpointers[t][best_last_tag]
        tags.append(index_to_tag[best_last_tag])

    return tags[::-1]

def generate_pseudo_word_data(train_data, test_data, threshold=5):
    word_counts = Counter(word for sentence in train_data for word, _ in sentence)
    rare_words = {word for word, count in word_counts.items() if count < threshold}

    def replace_with_tags(data):
        new_data = []
        for sentence in data:
            new_sentence = []
            for word, tag in sentence:
                if word in rare_words:
                    new_word = tag  # Replace rare words with their tag
                else:
                    new_word = word
                new_sentence.append((new_word, tag))
            new_data.append(new_sentence)
        return new_data

    return replace_with_tags(train_data), replace_with_tags(test_data)

def build_confusion_matrix(test_data, transition_probs, emission_probs, all_tags, most_likely_tag):
    y_true = []
    y_pred = []

    for sentence in test_data:
        words, true_tags = zip(*sentence)
        predicted_tags = viterbi(words, transition_probs, emission_probs, all_tags)
        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)

    tag_list = sorted(list(all_tags))
    cm = confusion_matrix(y_true, y_pred, labels=tag_list)
    return cm, tag_list

if __name__ == "__main__":
    tagged_sentences = download_and_load_corpus()
    train_data, test_data = split_corpus(tagged_sentences)

    most_likely_tag = compute_most_likely_tags(train_data)
    baseline_error_rate = evaluate_baseline(test_data, most_likely_tag)
    print("Baseline Error Rate:", baseline_error_rate)

    transition_probs, emission_probs, all_tags = compute_hmm_probabilities(train_data)
    hmm_error_rate = evaluate_hmm(test_data, transition_probs, emission_probs, all_tags, most_likely_tag)
    print("HMM Error Rate:", hmm_error_rate)

    smoothed_transition_probs, smoothed_emission_probs, smoothed_tags = add_one_smoothing(train_data)
    smoothed_error_rate = evaluate_hmm(test_data, smoothed_transition_probs, smoothed_emission_probs, smoothed_tags,
                                       most_likely_tag)
    print("Smoothed HMM Error Rate:", smoothed_error_rate)

    pseudo_train_data, pseudo_test_data = generate_pseudo_word_data(train_data, test_data)

    pseudo_transition_probs, pseudo_emission_probs, pseudo_tags = compute_hmm_probabilities(pseudo_train_data)
    pseudo_error_rate = evaluate_hmm(pseudo_test_data, pseudo_transition_probs, pseudo_emission_probs, pseudo_tags,
                                     most_likely_tag)
    print("Pseudo-Words HMM Error Rate:", pseudo_error_rate)

    smoothed_pseudo_transition_probs, smoothed_pseudo_emission_probs, smoothed_pseudo_tags = add_one_smoothing(
        pseudo_train_data)
    smoothed_pseudo_error_rate = evaluate_hmm(pseudo_test_data, smoothed_pseudo_transition_probs,
                                              smoothed_pseudo_emission_probs, smoothed_pseudo_tags, most_likely_tag)
    print("Smoothed Pseudo-Words HMM Error Rate:", smoothed_pseudo_error_rate)

    confusion_mat, tags = build_confusion_matrix(
        pseudo_test_data, smoothed_pseudo_transition_probs, smoothed_pseudo_emission_probs, smoothed_pseudo_tags,
        most_likely_tag
    )
    print("\nConfusion Matrix:")
    print(confusion_mat)

    most_frequent_errors = [
        (tags[i], tags[j], confusion_mat[i, j])
        for i in range(len(tags))
        for j in range(len(tags))
        if i != j and confusion_mat[i, j] > 0
    ]
    most_frequent_errors.sort(key=lambda x: x[2], reverse=True)

    print("\nMost Frequent Errors:")
    for true_tag, predicted_tag, count in most_frequent_errors[:10]:
        print(f"True Tag: {true_tag}, Predicted Tag: {predicted_tag}, Count: {count}")
