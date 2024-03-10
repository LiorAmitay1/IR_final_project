import math
# from collections import Counter, OrderedDict, defaultdict
import threading

import inflect
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from inverted_index_gcp import *
import multiprocessing

# import inflect

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
MAX_THREADS = 10
thread_pool = []
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

english_stopwords = frozenset(stopwords.words('english'))
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
RE_WORD2 = re.compile(r"""[\w\d]+(?:['\-]\w+){0,23}""", re.UNICODE)
question_words = ["what", "where", "who", "why", "how", "whom", "which", "whose", "when", "can", "could", "?"]

# def execute_search(query, res):
#     try:
#         # Call the retrieval function and append the results to the global res list
#         search_all(query, text_index,title_index, docid_count_dict_pkl, docid_title_dict_pkl, page_views_pkl, res)
#     except Exception as e:
#         logging.exception("Error occurred during search", e)
#     finally:
#         # Remove the thread from the pool after completion
#         thread_pool.remove(threading.current_thread())


"""That's all the functions for tokenization"""


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return tokens


def tokenize_word_with_numbers(text):
    tokens = [token.group() for token in RE_WORD2.finditer(text.lower()) if token.group() not in all_stopwords]
    return tokens


def tokenize_stemmer(text):
    stemmer = PorterStemmer()
    num_to_letters = convert_to_word(text)
    text_new = fined_special_letters(text)
    tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if
              token.group() not in all_stopwords]
    for word in text_new:
        tokens.append(word)
    for word in num_to_letters:
        tokens.append(word)
    return tokens

def check_special_characters(sentence):
    # Define a regular expression pattern to match special characters
    special_characters_pattern = r'[?\"\'\d]'

    # Check if the sentence contains any special characters
    if re.search(special_characters_pattern, sentence):
        return True
    else:
        return False


def looking_for_special(query):
    query = query.lower()
    tokens = []
    # if the query is question:
    for word in question_words:
        if word in query:
            isQuestion = True
            tokens = nltk_check(query, 2, isQuestion)
            break

    # to give more weight to the words that in the phrases
    phrases = re.findall(r'"([^"]+)"', query)
    if phrases:
        token_to_phrases = token_only_phrases(phrases)
        tokens += token_to_phrases

    # to check if there isn't word with number that we missed
    special_words = tokenize_word_with_numbers(query)
    for w in special_words:
        if w not in tokens:
            tokens.append(w)

    # to convert words that are numbers or symbols to words
    num_to_words = tokenize_with_unique(query)
    for w in num_to_words:
        if w not in tokens:
            tokens.append(w)
    return tokens


def tokenize_with_unique(text):
    num_to_letters = convert_to_word(text)
    text_new = fined_special_letters(text)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    for word in text_new:
        tokens.append(word)
    for word in num_to_letters:
        tokens.append(word)
    return tokens


def convert_to_word(tokens):
    p = inflect.engine()
    to_return = []
    for i, token in enumerate(tokens):
        if token:
            if token.isdigit():
                x = p.number_to_words(token)
                to_return.append(x)
    return to_return


def roman_to_arabic(roman):
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0

    for numeral in reversed(roman):
        value = roman_numerals[numeral]
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value

    return result


def fined_special_letters(text):
    words = text.split()
    converted_words = []
    for word in words:
        if word.upper() in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']:
            arabic_numeral = roman_to_arabic(word.upper())
            if arabic_numeral == 1:
                converted_words.append('one')
            elif arabic_numeral == 2:
                converted_words.append('two')
            elif arabic_numeral == 3:
                converted_words.append('three')
            elif arabic_numeral == 4:
                converted_words.append('four')
            elif arabic_numeral == 5:
                converted_words.append('five')
            elif arabic_numeral == 6:
                converted_words.append('six')
            elif arabic_numeral == 7:
                converted_words.append('seven')
            elif arabic_numeral == 8:
                converted_words.append('eight')
            elif arabic_numeral == 9:
                converted_words.append('nine')
    return converted_words





def token_only_phrases(phrases):
    # Extract phrases enclosed in double quotes
    new_query = ""
    for word in phrases:
        new_query += " " + word
    return tokenize(new_query)


def nltk_check(query, num=10, isQuestion=False):
    query_new = tokenize(query)
    query_final = []
    if isQuestion:
        try:
            pos_tags = nltk.pos_tag(query_new)
            keywords = [word for word, pos in pos_tags if (pos.startswith("NN") or pos.startswith("CD"))]
            query_final = query_new + keywords * num
        except:
            pass
    return query_final


"""This is all the functions in which we chose to take a certain number of documents"""


def get_top_n(dict1, n=50):
    # Sort the dictionary items by their values in descending order
    sorted_items = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
    top_100_keys = [item[0] for item in sorted_items[:n]]
    return top_100_keys


def get_top_n_in_dict(dict1, n=80):
    # Sort the dictionary by values in descending order
    sorted_dict = sorted(dict1.items(), key=lambda item: item[1], reverse=True)
    # Take the first 100 items
    top_100 = dict(sorted_dict[:n])
    return top_100


"""These are all the functions with which we combined the documents from the various calculations"""


def merge_results_of_title_and_text(dict_scores_weight):
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1]
    return merge_dict


def merge_results_of_title_and_text_with_pageView(dict_scores_weight, page_view_dict):
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1] * math.log(
                page_view_dict.get(doc_id, 0) + 2, 2)
    return merge_dict


def merge_results_of_title_and_text_with_PageRank(dict_scores_weight, page_rank_pkl):
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1] * math.log(page_rank_pkl.get(doc_id, 0) + 2,
                                                                                       2)
    return merge_dict


def merge_results_of_title_and_text_with_pageView_PageRank(dict_scores_weight, page_view_dict, page_rank_pkl):
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1] * math.log(
                page_view_dict.get(doc_id, 0) + 2, 2) * math.log((page_rank_pkl.get(doc_id, 0) + 2), 2)
    return merge_dict


def merge_results_of_title_and_text_with_PV_PR_anchor(dict_scores_weight, anchor_dict, page_view_dict, page_rank_pkl):
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            pr_pv_score = score * x[1] * math.log(page_view_dict.get(doc_id, 0) + 2, 2) * math.log(
                (page_rank_pkl.get(doc_id, 0) + 2), 2)
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + max(anchor_dict.get(doc_id, 0.0) * pr_pv_score,
                                                                   pr_pv_score)
    return merge_dict


def merge_results_of_title_and_text_with_PV_PR_anchor_precent(top_text, top_title, anchor_dict, page_view_dict,
                                                              page_rank_pkl, question=False):
    merge_dict = {}
    Wa = 0.95
    Wb = 1.1
    Wt = 0.95
    Wpv = 1
    Wpr = 1
    if question:
        Wb = Wb * 1.2
        Wa = Wa * 1.2
        Wpv = Wpv * 0.8
        Wpr = Wpr * 0.8

    for doc_id, score in top_text.items():
        if doc_id in merge_dict:
            merge_dict[doc_id] += score * Wb
        else:
            merge_dict[doc_id] = score * Wb

    for doc_id, score in top_title.items():
        if doc_id in merge_dict:
            merge_dict[doc_id] += score * Wt
        else:
            merge_dict[doc_id] = score * Wt

    for doc_id, score in anchor_dict.items():
        if doc_id in merge_dict:
            merge_dict[doc_id] += score * Wa
        else:
            merge_dict[doc_id] = score * Wa

    for doc_id, score in merge_dict.items():
        merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + Wpv * math.log(page_view_dict.get(doc_id, 0) + 2,
                                                                          2) + Wpr * math.log(
            (page_rank_pkl.get(doc_id, 0) + 2), 2)

    return merge_dict


def result_doc_to_title(arr, titles_dict):
    """
    Args:
        arr: array of doc_id
        titles_dict: dict that the key is doc id and value is the title
    Returns:
        array of tuples [(doc_id,title),(doc_id,title)....]
    """
    result = []
    for doc_id in arr:
        result.append((str(doc_id), titles_dict.get(doc_id, "Not found doc title")))
        # result.append((doc_id, titles_dict.get(doc_id, "Not found doc title")))
    return result


"""That's all the functions that the different retrieval models we've chosen have"""


def boolean_search(query, inverted):
    dict_return = defaultdict(float)
    for word in query:
        posting = inverted.read_a_posting_list('', word, 'bucket_318820123')
        for doc_id, count in posting:
            if doc_id in dict_return:
                dict_return[doc_id] += 1
            else:
                dict_return[doc_id] = 1

            # dict_return[doc_id] = dict_return.get(doc_id, 0.0) + 1
    return dict_return


def BM25_search(query, title_index):
    # query = tokenize(old_query)
    word_repeat = Counter(query)
    size_query = len(query)
    dict_return = {}
    for word in query:
        posting = title_index.read_a_posting_list('', word, 'bucket_318820123')
        f = word_repeat[word]
        tf_query = f / size_query
        for doc_id, x in posting:
            sum = 0
            dict_return[doc_id] = 0
            tf_doc = x / title_index.dict_docID_countWords[doc_id]
            idf = math.log10((title_index.num_of_doc + 1) / title_index.df[word])
            k1 = 1.2
            k3 = 1.5
            b = 0.25
            B = 1 - b + (b * (title_index.dict_docID_countWords[doc_id] / title_index.avg))
            sum += ((k1 + 1) * tf_doc / (B * k1 + tf_doc)) * idf * ((k3 + 1) * tf_query / (k3 + tf_query))
            dict_return[doc_id] += sum
    return dict_return


def BM25_search_text(query, title_index, docID_count_dict, k1=1.2, k3=1.5):
    word_repeat = Counter(query)
    size_query = len(query)
    dict_return = {}
    for word in query:
        posting = title_index.read_a_posting_list("", word, 'bucket_318820123')
        f = word_repeat[word]
        tf_query = f / size_query
        for doc_id, x in posting:
            sum = 0
            dict_return[doc_id] = 0
            tf_doc = x / docID_count_dict.get(doc_id, 1)
            idf = math.log10((title_index.num_of_doc + 1) / title_index.df[word])
            b = 0.25
            B = 1 - b + (b * (docID_count_dict.get(doc_id, 1) / title_index.avg))
            sum += ((k1 + 1) * tf_doc / (B * k1 + tf_doc)) * idf * ((k3 + 1) * tf_query / (k3 + tf_query))
            dict_return[doc_id] += sum
    return dict_return


def cosine_similarity_search(query, inverted, normalization_freq_dict):
    # query = tokenize(old_query)
    word_repeat = Counter(query)
    num_of_doc = inverted.num_of_doc
    dict_return = defaultdict(float)
    for word in query:
        posting = inverted.read_a_posting_list('', word, 'bucket_318820123')
        w_df = inverted.df.get(word, 1)
        w_qtf = word_repeat.get(word, 1)
        idf = math.log(num_of_doc / w_df, 10)
        for doc_id, freq in posting:
            dict_return[doc_id] += freq * idf * w_qtf
        Qnorm = math.sqrt(sum([tf ** 2 for tf in word_repeat.values()]))
        for doc_id in dict_return.keys():
            dict_return[doc_id] = dict_return[doc_id] * (1 / Qnorm) * normalization_freq_dict.get(doc_id, 0.1)
    return dict_return


class Search_with_thread():
    def __init__(self):
        self.res_title = []
        self.res_text = []
        self.lock = threading.Lock()

    def search_thread(self, query_text, query_without_stem, text_index, title_index, docid_count_dict_pkl, normalization_freq_dict_pkl = None):
        if normalization_freq_dict_pkl is not None:
            title = threading.Thread(target=self.search_title_bm25, args=(query_without_stem, title_index,))
            text = threading.Thread(target=self.search_text_cosine_sim, args=(query_text, text_index, normalization_freq_dict_pkl,))

        else:
            title = threading.Thread(target=self.search_title_boolean, args=(query_without_stem, title_index,))
            text = threading.Thread(target=self.search_text_bm25,
                                    args=(query_text, text_index, docid_count_dict_pkl,))
        title.start()
        text.start()
        title.join()
        text.join()
        return_title = self.res_title
        return_text = self.res_text
        return return_title, return_text

    def search_title_boolean(self, query, title_index):
        search_for_title = boolean_search(query, title_index)
        top_title = get_top_n_in_dict(search_for_title)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_title = top_title
        return

    def search_title_bm25(self, query, title_index):
        search_for_title = BM25_search(query, title_index)
        top_title = get_top_n_in_dict(search_for_title)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_title = top_title
        return

    def search_text_bm25(self, query, text_index, docid_count_dict_pkl):
        # query = nltk_check_parallel(query, 5)
        search_for_text = BM25_search_text(query, text_index, docid_count_dict_pkl, 1.5, 2.5)
        top_text = get_top_n_in_dict(search_for_text)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_text = top_text
        return

    def search_text_cosine_sim(self, query, text_index, normalization_freq_dict_pkl):
        # query = nltk_check_parallel(query, 5)
        search_for_text = cosine_similarity_search(query, text_index, normalization_freq_dict_pkl)
        top_text = get_top_n_in_dict(search_for_text)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_text = top_text
        return

############################################################################
# These are the functions we enjoyed and did not work for us as we expected


# def nltk_check_parallel(query, num=10):
#     # Extract phrases enclosed in double quotes
#     phrases = re.findall(r'"([^"]+)"', query)
#     for phrase in phrases:
#         print(phrase)
#         query = query.replace('"' + phrase + '"', phrase)  # Remove the phrase from the query
#
#     # Tokenize the remaining query terms, preserving numbers as separate tokens
#     tokens = tokenize(query)
#     print(query)
#     # Give more weight to the words within the extracted phrases
#     weighted_tokens = []
#
#     for token in tokens:
#         if any(word in token for word in phrases):
#             weighted_tokens.extend([token] * (num + 5))  # Give a higher weight to words within phrases
#         elif token[0].isupper() and token.lower() not in question_words:
#             weighted_tokens.extend([token] * (num + 3))  # Give a higher weight to words starting with a capital letter
#         else:
#             weighted_tokens.append(token.lower())  # Convert the token to lowercase and append it
#
#     # Process the query as usual
#     isQuestion = any(word in query for word in question_words)
#     if isQuestion:
#         try:
#             with multiprocessing.Pool() as pool:
#                 pos_tags = pool.map(nltk.pos_tag, [weighted_tokens])[0]
#             keywords = [word for word, pos in pos_tags if (pos.startswith("NN"))]
#             weighted_tokens += keywords * num
#         except Exception as e:
#             print("Error:", e)
#             pass
#     lower_token = [word.lower() for word in weighted_tokens]
#     final_q = [w for w in lower_token if w not in question_words]
#     return final_q


# def nltk_check_parallel(query, num=10):
#     # Extract phrases enclosed in double quotes
#     phrases = re.findall(r'"([^"]+)"', query)
#     for phrase in phrases:
#         query = query.replace('"' + phrase + '"', '')  # Remove the phrase from the query
#
#     # Tokenize the remaining query terms, preserving numbers as separate tokens
#     tokens = [token.group() for token in RE_WORD.finditer(query)]
#
#     # Give more weight to the words within the extracted phrases
#     weighted_tokens = []
#     for token in tokens:
#         if any(word in token for word in phrases):
#             weighted_tokens.extend([token] * (num + 5))  # Give a higher weight to words within phrases
#         elif token[0].isupper() and token.lower() not in question_words:
#             weighted_tokens.extend([token] * (num + 3))  # Give a higher weight to words starting with a capital letter
#         else:
#             weighted_tokens.append(token.lower())  # Convert the token to lowercase and append it
#
#     # Process the query as usual
#     isQuestion = any(word in query for word in question_words)
#     if isQuestion:
#         try:
#             with multiprocessing.Pool() as pool:
#                 pos_tags = pool.map(nltk.pos_tag, [weighted_tokens])[0]
#             keywords = [word for word, pos in pos_tags if pos.startswith("NN")]
#             weighted_tokens += keywords * num
#         except Exception as e:
#             print("Error:", e)
#             pass
#
#     return weighted_tokens


# def nltk_check(query, num=10, isQuestion=False):
#     query = query.lower()
#     for word in question_words:
#         if word in query:
#             isQuestion = True
#             break
#
#     query_new = tokenize(query)
#     # print(query_new)
#     query_final = convert_to_word(query_new)
#     # print(query_final)
#     if isQuestion:
#         try:
#             pos_tags = nltk.pos_tag(query_final)
#             keywords = [word for word, pos in pos_tags if (pos.startswith("NN") or pos.startswith("CD"))]
#             # print(pos_tags)
#             query_final = query_final + keywords * num
#         except:
#             pass
#     return query_final
#


# def nltk_check_parallel(query, num=10):
#     # Extract phrases enclosed in double quotes
#     phrases = re.findall(r'"([^"]+)"', query)
#     for phrase in phrases:
#         query = query.replace('"' + phrase + '"', '')  # Remove the phrase from the query
#
#     # Tokenize the remaining query terms, preserving numbers as separate tokens
#     tokens = [token.group() for token in RE_WORD.finditer(query)]
#
#     # Convert numbers to words
#     p = inflect.engine()
#     for i, token in enumerate(tokens):
#         if token.isdigit():
#             tokens[i] = p.number_to_words(token)
#
#     # Give more weight to the words within the extracted phrases
#     weighted_tokens = []
#     for token in tokens:
#         if any(word in token for word in phrases):
#             weighted_tokens.extend([token] * (num + 5))  # Give a higher weight to words within phrases
#         elif token[0].isupper() and token.lower() not in question_words:
#             weighted_tokens.extend([token] * (num + 3))  # Give a higher weight to words starting with a capital letter
#         else:
#             weighted_tokens.append(token.lower())  # Convert the token to lowercase and append it
#
#     # Process the query as usual
#     # isQuestion = any(word in query for word in question_words)
#     # if isQuestion:
#     #     try:
#     #         with multiprocessing.Pool() as pool:
#     #             pos_tags = pool.map(nltk.pos_tag, [weighted_tokens])[0]
#     #         keywords = [word for word, pos in pos_tags if pos.startswith("NN")]
#     #         weighted_tokens += keywords * num
#     #     except Exception as e:
#     #         print("Error:", e)
#     #         # pass
#
#     return weighted_tokens


# attemption to get more correct querys but didnt worked:
# def ner(query):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(query)
#     named_entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return named_entities
