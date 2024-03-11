import math
import threading
import inflect
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from inverted_index_gcp import *


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


"""That's all the functions for tokenization"""


def tokenize(text):
    """
    Tokenizes the input text into words, converting them to lowercase and filtering out stopwords.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list: A list of tokens (words) extracted from the input text.
    """
    # Tokenize the input text, convert to lowercase, and filter out stopwords
    tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return tokens


def tokenize_word_with_numbers(text):
    """
    Tokenizes the input text into words, converting them to lowercase and filtering out stopwords.

    Parameters:
    text (str): The input text to be tokenized. with the numbers and letters

    Returns:
    list: A list of tokens (words) extracted from the input text.
    """
    # Tokenize the input text, convert to lowercase, and filter out stopwords
    tokens = [token.group() for token in RE_WORD2.finditer(text.lower()) if token.group() not in all_stopwords]
    return tokens


def tokenize_stemmer(text):
    """
    Tokenizes the input text, applies stemming, and includes special letters and numbers as tokens.
    Parameters:
        text (str): The input text to be tokenized.
    Returns:
        list of str: A list of tokens after tokenization, stemming, and inclusion of special characters and numbers.
    Note:
        This function uses the Porter Stemmer algorithm to reduce words to their root form.
        Special letters and numbers are added back into the token list after stemming.

    """
    stemmer = PorterStemmer()
    # Function to convert numbers to words
    num_to_letters = convert_to_word(text)
    # Function to handle special characters
    text_new = fined_special_letters(text)
    # Tokenize and stem words
    tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if
              token.group() not in all_stopwords]
    for word in text_new:
        # Add special letters back into tokens
        tokens.append(word)
    for word in num_to_letters:
        # Add converted numbers back into tokens
        tokens.append(word)
    return tokens

def check_special_characters(sentence):
    """
    Checks if a given sentence contains any special characters.

    Parameters:
        sentence (str): The input sentence to be checked for special characters.

    Returns:
        bool: True if the sentence contains special characters, False otherwise.
    """
    # Define a regular expression pattern to match special characters
    special_characters_pattern = r'[?\"\'\d]'

    # Check if the sentence contains any special characters
    if re.search(special_characters_pattern, sentence):
        return True
    else:
        return False


def looking_for_special(query):
    """
    Tokenizes a query while considering special cases such as question words, phrases in quotes,
    words with numbers, and converting numbers/symbols to words.
    Args:
        query (str): The input query to be tokenized.
    Returns:
        list: A list of tokens extracted from the query.
    """
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
    """
    Tokenizes the input text into unique words, including special characters and numbers converted to words.

    Parameters:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of unique tokens extracted from the input text.

    Note:
        This function applies lowercase conversion to the text before tokenization.
        It also removes stopwords defined in the global variable 'all_stopwords'.
        Additionally, it converts special characters and numbers to words and includes them in the token list.
    """
    # Convert numbers to words
    num_to_letters = convert_to_word(text)
    # Find special letters and characters
    text_new = fined_special_letters(text)
    # Tokenize the text into words, excluding stopwords
    tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    # Add special letters and characters to the token list
    for word in text_new:
        tokens.append(word)
    # Add converted numbers to words to the token list
    for word in num_to_letters:
        tokens.append(word)
    return tokens


def convert_to_word(tokens):
    """
    Convert numerical tokens to words using the inflect library.

    Parameters:
    tokens (list of str): A list of tokens to be converted. Each token can be a numerical value represented as a string.

    Returns:
    list of str: A list of tokens where numerical values have been converted to words.
    """
    p = inflect.engine()
    to_return = []
    for i, token in enumerate(tokens):
        if token:
            if token.isdigit():
                x = p.number_to_words(token)
                to_return.append(x)
    return to_return


def roman_to_arabic(roman):
    """
    Convert a Roman numeral to an Arabic numeral.

    Parameters:
    roman (str): The Roman numeral to be converted.

    Returns:
    int: The equivalent Arabic numeral.

    Raises:
    ValueError: If the input string contains invalid Roman numerals.

    Example:
    roman_to_arabic('III')
    3
    roman_to_arabic('XLII')
    42
    """
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
    """
        Converts special letter sequences such as Roman numerals into their word equivalents.

        Parameters:
        text (str): The input text containing special letter sequences.

        Returns:
        list of str: A list of words with special letter sequences replaced by their word equivalents.
        """
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
    """
    Tokenizes only the phrases enclosed in double quotes from a list of phrases.

    Parameters:
    phrases (list of str): A list of phrases.

    Returns:
    list of str: A list of tokens extracted from the phrases enclosed in double quotes.
    """
    # Extract phrases enclosed in double quotes
    new_query = ""
    for word in phrases:
        new_query += " " + word
    return tokenize(new_query)


def nltk_check(query, num=10, isQuestion=False):
    """
    Generates multiple queries based on the input query by tokenizing and adding keywords.

    Parameters:
    query (str): The input query string.
    num (int): The number of additional queries to generate (default is 10).
    isQuestion (bool): Flag indicating whether the input query is a question (default is False).

    Returns:
    list of str: A list of generated queries based on the input query and specified parameters.
    """
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
    """
    Retrieves the top N keys from a dictionary based on their corresponding values.

    Parameters:
    dict1 (dict): The input dictionary.
    n (int): The number of top keys to retrieve (default is 50).

    Returns:
    list: A list containing the top N keys from the input dictionary, sorted by their values in descending order.
    """
    # Sort the dictionary items by their values in descending order
    sorted_items = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
    # Extract the top N keys
    top_n_keys = [item[0] for item in sorted_items[:n]]
    return top_n_keys


def get_top_n_in_dict(dict1, n=80):
    """
    Retrieves the top N key-value pairs from a dictionary based on their values.

    Parameters:
    dict1 (dict): The input dictionary.
    n (int): The number of top key-value pairs to retrieve (default is 80).

    Returns:
    dict: A dictionary containing the top N key-value pairs from the input dictionary, sorted by values in descending order.
    """
    # Sort the dictionary by values in descending order
    sorted_dict = sorted(dict1.items(), key=lambda item: item[1], reverse=True)
    # Take the first n items
    top_n = dict(sorted_dict[:n])
    return top_n


"""These are all the functions with which we combined the documents from the various calculations"""


def merge_results_of_title_and_text(dict_scores_weight):
    """
    Merges the scores of documents obtained from title and text search with their corresponding weights.

    Parameters:
    dict_scores_weight (list of tuple): A list of tuples containing dictionaries of document scores and their corresponding weights.

    Returns:
    dict: A dictionary containing merged document scores weighted by their respective weights.
    """
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1]
    return merge_dict


def merge_results_of_title_and_text_with_pageView(dict_scores_weight, page_view_dict):
    """
    Merges the scores of documents obtained from title and text search with their corresponding weights,
    taking into account the page views of each document.

    Parameters:
    dict_scores_weight (list of tuple): A list of tuples containing dictionaries of document scores and their corresponding weights.
    page_view_dict (dict): A dictionary containing the page views of each document.

    Returns:
    dict: A dictionary containing merged document scores weighted by their respective weights and adjusted by page views.
    """
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1] * math.log(
                page_view_dict.get(doc_id, 0) + 2, 2)
    return merge_dict


def merge_results_of_title_and_text_with_PageRank(dict_scores_weight, page_rank_pkl):
    """
    Merges the scores of documents obtained from title and text search with their corresponding weights,
    taking into account the PageRank scores of each document.

    Parameters:
    dict_scores_weight (list of tuple): A list of tuples containing dictionaries of document scores and their corresponding weights.
    page_rank_pkl (dict): A dictionary containing the PageRank scores of each document.

    Returns:
    dict: A dictionary containing merged document scores weighted by their respective weights and adjusted by PageRank scores.
    """
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1] * math.log(page_rank_pkl.get(doc_id, 0) + 2,
                                                                                       2)
    return merge_dict


def merge_results_of_title_and_text_with_pageView_PageRank(dict_scores_weight, page_view_dict, page_rank_pkl):
    """
    Merges the scores of documents obtained from title and text search with their corresponding weights,
    taking into account both the page views and PageRank scores of each document.

    Parameters:
    dict_scores_weight (list of tuple): A list of tuples containing dictionaries of document scores and their corresponding weights.
    page_view_dict (dict): A dictionary containing the page views of each document.
    page_rank_pkl (dict): A dictionary containing the PageRank scores of each document.

    Returns:
    dict: A dictionary containing merged document scores weighted by their respective weights, adjusted by both page views and PageRank scores.
    """
    merge_dict = {}
    for x in dict_scores_weight:
        for doc_id, score in x[0].items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * x[1] * math.log(
                page_view_dict.get(doc_id, 0) + 2, 2) * math.log((page_rank_pkl.get(doc_id, 0) + 2), 2)
    return merge_dict


def merge_results_of_title_and_text_with_PV_PR_anchor(dict_scores_weight, anchor_dict, page_view_dict, page_rank_pkl):
    """
    Merges the scores of documents obtained from title and text search with their corresponding weights,
    taking into account page views, PageRank scores, and anchor text relevance.

    Parameters:
    dict_scores_weight (list of tuple): A list of tuples containing dictionaries of document scores and their corresponding weights.
    anchor_dict (dict): A dictionary containing anchor text relevance scores for each document.
    page_view_dict (dict): A dictionary containing the page views of each document.
    page_rank_pkl (dict): A dictionary containing the PageRank scores of each document.

    Returns:
    dict: A dictionary containing merged document scores weighted by their respective weights,
    adjusted by page views, PageRank scores, and anchor text relevance.
    """
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
    """
    Merges scores from title, text, anchor text, page views, and PageRank, with customizable weights and adjustments.

    Parameters:
    top_text (dict): Dictionary containing document scores from text search.
    top_title (dict): Dictionary containing document scores from title search.
    anchor_dict (dict): Dictionary containing anchor text relevance scores for each document.
    page_view_dict (dict): Dictionary containing the page views of each document.
    page_rank_pkl (dict): Dictionary containing the PageRank scores of each document.
    question (bool): Flag indicating whether the search query is a question (default is False).

    Returns:
    dict: A dictionary containing merged document scores weighted by specified factors.
    """
    merge_dict = {}
    weight_anchor = 0.9  # Default weight for anchor text relevance
    weight_body = 0.5  # Default weight for text search results
    weight_title = 0.95  # Default weight for title search results
    weight_page_view = 1  # Default weight for page views
    weight_page_rank = 1  # Default weight for PageRank scores

    # Adjust weights if the search query is a question
    if question:
        weight_body = weight_body * 1.2
        weight_anchor = weight_anchor * 1.2
        weight_page_view = weight_page_view * 0.8
        weight_page_rank = weight_page_rank * 0.8

    # Merge scores from text search results
    for doc_id, score in top_text.items():
        if doc_id in merge_dict:
            merge_dict[doc_id] += score * weight_body
        else:
            merge_dict[doc_id] = score * weight_body

    # Merge scores from title search results
    for doc_id, score in top_title.items():
        if doc_id in merge_dict:
            merge_dict[doc_id] += score * weight_title
        else:
            merge_dict[doc_id] = score * weight_title

    # Merge scores from anchor text relevance
    for doc_id, score in anchor_dict.items():
        if doc_id in merge_dict:
            merge_dict[doc_id] += score * weight_anchor
        else:
            merge_dict[doc_id] = score * weight_anchor

    # Adjust merged scores by page views and PageRank
    for doc_id, score in merge_dict.items():
        merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + weight_page_view * math.log(page_view_dict.get(doc_id, 0) + 2,
                                                                          2) + weight_page_rank * math.log(
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
    return result


"""That's all the functions that the different retrieval models we've chosen have"""


def boolean_search(query, inverted):
    """
    Performs boolean search on an inverted index to retrieve documents containing all the terms in the query.

    Parameters:
    query (list of str): The list of terms to search for.
    inverted (InvertedIndex): The inverted index object containing the index data.

    Returns:
    dict: A dictionary containing document IDs as keys and their corresponding relevance scores based on term frequency.
    """
    dict_return = defaultdict(float)
    for word in query:
        posting = inverted.read_a_posting_list('', word, 'bucket_318820123')
        for doc_id, count in posting:
            if doc_id in dict_return:
                dict_return[doc_id] += 1
            else:
                dict_return[doc_id] = 1
    return dict_return


def BM25_search(query, inverted_index):
    """
    Performs BM25 search on a title index to retrieve documents relevant to the given query.

    Parameters:
    query (list of str): The list of terms in the query.
    title_index (TitleIndex): The title index object containing the index data.

    Returns:
    dict: A dictionary containing document IDs as keys and their corresponding relevance scores based on BM25 algorithm.
    """
    word_repeat = Counter(query)
    size_query = len(query)
    dict_return = {}
    for word in query:
        posting = inverted_index.read_a_posting_list('', word, 'bucket_318820123')
        f = word_repeat[word]
        tf_query = f / size_query
        for doc_id, x in posting:
            sum = 0
            dict_return[doc_id] = 0
            tf_doc = x / inverted_index.dict_docID_countWords[doc_id]
            idf = math.log10((inverted_index.num_of_doc + 1) / inverted_index.df[word])
            k1 = 1.2
            k3 = 1.5
            b = 0.25
            B = 1 - b + (b * (inverted_index.dict_docID_countWords[doc_id] / inverted_index.avg))
            sum += ((k1 + 1) * tf_doc / (B * k1 + tf_doc)) * idf * ((k3 + 1) * tf_query / (k3 + tf_query))
            dict_return[doc_id] += sum
    return dict_return


def BM25_search_text(query, inverted_index, docID_count_dict, k1=1.2, k3=1.5):
    """
    Performs BM25 search on an inverted index to retrieve documents relevant to the given query,
    considering the text content of the documents.

    Parameters:
    query (list of str): The list of terms in the query.
    inverted_index (InvertedIndex): The inverted index object containing the index data.
    docID_count_dict (dict): A dictionary containing the word count of each document.
    k1 (float): A tuning parameter controlling term frequency normalization (default is 1.2).
    k3 (float): A tuning parameter controlling query term normalization (default is 1.5).

    Returns:
    dict: A dictionary containing document IDs as keys and their corresponding relevance scores based on BM25 algorithm.
    """
    word_repeat = Counter(query)
    size_query = len(query)
    dict_return = {}
    for word in query:
        posting = inverted_index.read_a_posting_list("", word, 'bucket_318820123')
        f = word_repeat[word]
        tf_query = f / size_query
        for doc_id, x in posting:
            sum = 0
            dict_return[doc_id] = 0
            tf_doc = x / docID_count_dict.get(doc_id, 1)
            idf = math.log10((inverted_index.num_of_doc + 1) / inverted_index.df[word])
            b = 0.25
            B = 1 - b + (b * (docID_count_dict.get(doc_id, 1) / inverted_index.avg))
            sum += ((k1 + 1) * tf_doc / (B * k1 + tf_doc)) * idf * ((k3 + 1) * tf_query / (k3 + tf_query))
            dict_return[doc_id] += sum
    return dict_return


def cosine_similarity_search(query, inverted, normalization_freq_dict):
    """
    Performs cosine similarity search on an inverted index to retrieve documents relevant to the given query.

    Parameters:
    query (list of str): The list of terms in the query.
    inverted (InvertedIndex): The inverted index object containing the index data.
    normalization_freq_dict (dict): A dictionary containing the normalization frequency of each document.

    Returns:
    dict: A dictionary containing document IDs as keys and their corresponding cosine similarity scores with the query.
    """
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
    """
    Class for performing parallel search operations with threads.

    Attributes:
    res_title (list): List to store search results for titles.
    res_text (list): List to store search results for text.
    lock (threading.Lock): Lock for thread-safe access to shared data.
    """

    def __init__(self):
        """
        Initializes the Search_with_thread object.
        """
        self.res_title = []
        self.res_text = []
        self.lock = threading.Lock()

    def search_thread(self, query_text, query_without_stem, text_index, title_index, docid_count_dict_pkl, normalization_freq_dict_pkl = None):
        """
        Performs parallel search operations for title and text.

        Parameters:
        query_text (list of str): The list of terms in the query for text search.
        query_without_stem (list of str): The list of terms in the query for title search.
        text_index (InvertedIndex): The text index object containing the index data.
        title_index (InvertedIndex): The title index object containing the index data.
        docid_count_dict_pkl (dict): A dictionary containing the word count of each document for text search.
        normalization_freq_dict_pkl (dict): A dictionary containing the normalization frequency of each document for text search (default is None).

        Returns:
        tuple: A tuple containing search results for titles and text.
        """
        if normalization_freq_dict_pkl is not None:
            title = threading.Thread(target=self.search_title_bm25, args=(query_without_stem, title_index,))
            text = threading.Thread(target=self.search_text_cosine_sim, args=(query_text, text_index, normalization_freq_dict_pkl,))

        else:
            title = threading.Thread(target=self.search_title_boolean, args=(query_text, title_index,))
            text = threading.Thread(target=self.search_text_bm25,
                                    args=(query_without_stem, text_index, docid_count_dict_pkl,))
        title.start()
        text.start()
        title.join()
        text.join()
        return_title = self.res_title
        return_text = self.res_text
        return return_title, return_text

    def search_title_boolean(self, query, title_index):
        """
        Performs boolean search for titles and stores the results.

        Parameters:
        query (list of str): The list of terms in the query.
        title_index (InvertedIndex): The title index object containing the index data.
        """
        search_for_title = boolean_search(query, title_index)
        top_title = get_top_n_in_dict(search_for_title)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_title = top_title
        return

    def search_title_bm25(self, query, title_index):
        """
        Performs BM25 search for titles and stores the results.

        Parameters:
        query (list of str): The list of terms in the query.
        title_index (InvertedIndex): The title index object containing the index data.
        """
        search_for_title = BM25_search(query, title_index)
        top_title = get_top_n_in_dict(search_for_title)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_title = top_title
        return

    def search_text_bm25(self, query, text_index, docid_count_dict_pkl):
        """
        Performs BM25 search for text and stores the results.

        Parameters:
        query (list of str): The list of terms in the query.
        text_index (InvertedIndex): The text index object containing the index data.
        docid_count_dict_pkl (dict): A dictionary containing the word count of each document.
        """
        search_for_text = BM25_search_text(query, text_index, docid_count_dict_pkl, 1.5, 2.5)
        top_text = get_top_n_in_dict(search_for_text)
        with self.lock:  # Acquire the lock before modifying shared data
            self.res_text = top_text
        return

    def search_text_cosine_sim(self, query, text_index, normalization_freq_dict_pkl):
        """
        Performs cosine similarity search for text and stores the results.

        Parameters:
        query (list of str): The list of terms in the query.
        text_index (InvertedIndex): The text index object containing the index data.
        normalization_freq_dict_pkl (dict): A dictionary containing the normalization frequency of each document.

        """
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



