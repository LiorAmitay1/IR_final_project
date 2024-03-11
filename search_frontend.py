from flask import Flask, request, jsonify
import retrieval
from inverted_index_gcp import *
from google.cloud import storage
import logging

logging.basicConfig(level=logging.DEBUG)

bucket_name = "bucket_318820123"


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def load_pkl(file_path1):
    '''
    Loads a pickled file from Google Cloud Storage.

    Parameters:
    file_path1 (str): The path to the pickled file in the storage bucket.

    Returns:
    --------
    obj: The object loaded from the pickled file.
    '''
    storage_client1 = storage.Client()
    bucket1 = storage_client1.bucket("bucket_318820123")
    blob1 = bucket1.blob(file_path1)
    contents1 = blob1.download_as_bytes()
    return pickle.loads(contents1)


def init():
    """
    This function initializes all dictionaries required to perform a search as a global var
    :return: none
    """
    global title_index
    global text_index
    global title_index_stemming
    global text_index_stemming
    global anchor_index

    global docid_title_dict_pkl
    global docid_count_dict_pkl
    global page_views_pkl
    global page_rank_pkl
    global normalization_freq_dict_pkl

    title_index = load_pkl('title_index.pkl')
    text_index = load_pkl('text_index.pkl')
    title_index_stemming = load_pkl('title_index_stemming.pkl')
    text_index_stemming = load_pkl('text_index_stemming.pkl')
    anchor_index = load_pkl('anchor_index.pkl')

    docid_title_dict_pkl = load_pkl('id_title_dict.pkl')
    docid_count_dict_pkl = load_pkl('dict_docID_countWords.pkl')
    page_views_pkl = load_pkl('pageviews-202108-user.pkl')
    page_rank_pkl = load_pkl('PageRank_page_rank_dict.pkl')
    normalization_freq_dict_pkl = load_pkl('dict_cosineSim_Normal.pkl')


@app.route("/search")
def search():
    ''' The function searches the query by BM25 for the text and Boolean for the title

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    try:
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        num_text_precision = 0.677
        num_title_precision = 0.323

        query_without_stem = retrieval.tokenize(query)
        if len(query_without_stem) <= 2:
            search_for_title = retrieval.boolean_search(query_without_stem, title_index)
            top_title = retrieval.get_top_n_in_dict(search_for_title)

            search_for_text = retrieval.BM25_search_text(query_without_stem, text_index, docid_count_dict_pkl, 1.5, 2.5)
            top_text = retrieval.get_top_n_in_dict(search_for_text)

            num_title_precision = 0.95
            num_text_precision = 0.05
        else:
            thread_search = retrieval.Search_with_thread()
            top_title, top_text = thread_search.search_thread(query_without_stem, query_without_stem, text_index,
                                                              title_index,
                                                              docid_count_dict_pkl)  # normalization_freq_dict_pkl)

        list_new = [(top_text, num_text_precision), (top_title, num_title_precision)]
        after_merge = retrieval.merge_results_of_title_and_text_with_pageView_PageRank(list_new, page_views_pkl,
                                                                                       page_rank_pkl)
        top_100_merge = retrieval.get_top_n(after_merge)
        res = retrieval.result_doc_to_title(top_100_merge, docid_title_dict_pkl)
        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        logging.exception("error", e)
        respose = ('error', str(e))
        return jsonify(respose)


"""Here are additional attempts for search that we have further developed or that we have chosen not to use"""


@app.route("/search_new")
def search_new():
    ''' The function searches the query by BM25 for the text and Boolean for the title with stemming for the title
    index, also search the specipic letters the in query

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    try:
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        num_text_precision = 0.577
        num_title_precision = 0.423
        thread_search = retrieval.Search_with_thread()
        thread_search1 = retrieval.Search_with_thread()
        query_stemmed = retrieval.tokenize_stemmer(query)

        if retrieval.check_special_characters(query):
            query_new1 = retrieval.looking_for_special(query)
            top_title, top_text = thread_search.search_thread(query_stemmed, query_new1, text_index,
                                                              title_index_stemming,
                                                              docid_count_dict_pkl)
        else:
            query_without_stem = retrieval.tokenize(query)
            if len(query_without_stem) <= 2:
                search_for_title = retrieval.boolean_search(query_stemmed, title_index_stemming)
                top_title = retrieval.get_top_n_in_dict(search_for_title)

                search_for_text = retrieval.BM25_search_text(query_without_stem, text_index, docid_count_dict_pkl, 1.5,
                                                             2.5)
                top_text = retrieval.get_top_n_in_dict(search_for_text)

                num_title_precision = 0.95
                num_text_precision = 0.05
            else:
                top_title, top_text = thread_search1.search_thread(query_stemmed, query_without_stem, text_index,
                                                                   title_index_stemming, docid_count_dict_pkl)

        list_new = [(top_text, num_text_precision), (top_title, num_title_precision)]
        after_merge = retrieval.merge_results_of_title_and_text_with_pageView_PageRank(list_new, page_views_pkl,
                                                                                       page_rank_pkl)
        top_100_merge = retrieval.get_top_n(after_merge)
        res = retrieval.result_doc_to_title(top_100_merge, docid_title_dict_pkl)
        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        logging.exception("error", e)
        respose = ('error', str(e))
        return jsonify(respose)

@app.route("/search_2ways")
def search_2ways():
    '''
    The function performs a search based on the provided query. The search is done according to several options,
    if there are special characters then search by boolean on the title and cosine similarity on the text. If there
    are no special characters then search by boolean on title and bm25 on the text

    To issue a query, navigate to a URL like:
    http://YOUR_SERVER_DOMAIN/search_2ways?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.

    Returns:
    --------
    list: A list of search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    try:
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        num_text_precision = 0.677
        num_title_precision = 0.323
        thread_search = retrieval.Search_with_thread()
        if retrieval.check_special_characters(query):
            query_new1 = retrieval.looking_for_special(query)
            top_title, top_text = thread_search.search_thread(query_new1, query_new1, text_index,
                                                              title_index,
                                                              docid_count_dict_pkl, normalization_freq_dict_pkl)
        else:
            query_without_stem = retrieval.tokenize(query)
            if len(query_without_stem) <= 2:
                search_for_title = retrieval.boolean_search(query_without_stem, title_index)
                top_title = retrieval.get_top_n_in_dict(search_for_title)

                search_for_text = retrieval.BM25_search_text(query_without_stem, text_index, docid_count_dict_pkl, 1.5,
                                                             2.5)
                top_text = retrieval.get_top_n_in_dict(search_for_text)

                num_title_precision = 0.95
                num_text_precision = 0.05
            else:
                top_title, top_text = thread_search.search_thread(query_without_stem, query_without_stem, text_index,
                                                                  title_index,
                                                                  docid_count_dict_pkl)

        list_new = [(top_text, num_text_precision), (top_title, num_title_precision)]
        after_merge = retrieval.merge_results_of_title_and_text_with_pageView_PageRank(list_new, page_views_pkl,
                                                                                       page_rank_pkl)
        top_100_merge = retrieval.get_top_n(after_merge)
        res = retrieval.result_doc_to_title(top_100_merge, docid_title_dict_pkl)
        return jsonify(res)
    except Exception as e:
        logging.exception("error", e)
        respose = ('error', str(e))
        return jsonify(respose)


@app.route("/search_with_stemming_text")
def search_with_stemming_text():
    ''' The function searches the query by BM25 for the text and Boolean for the title with stemming for the text
    index, also search the specipic letters the in query

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    try:
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        num_text_precision = 0.677
        num_title_precision = 0.323
        query_without_stem = retrieval.tokenize(query)
        query_text = retrieval.tokenize_stemmer(query)

        if len(query_without_stem) <= 2:
            search_for_title = retrieval.boolean_search(query_without_stem, title_index)
            top_title = retrieval.get_top_n_in_dict(search_for_title)
            search_for_text = retrieval.BM25_search_text(query_text, text_index_stemming, docid_count_dict_pkl, 1.5,
                                                         2.5)
            top_text = retrieval.get_top_n_in_dict(search_for_text)

            num_title_precision = 1
            num_text_precision = 0
        else:
            thread_search = retrieval.Search_with_thread()
            top_title, top_text = thread_search.search_thread(query_text, query_without_stem, text_index_stemming,
                                                              title_index,
                                                              docid_count_dict_pkl)

        list_new = [(top_text, num_text_precision), (top_title, num_title_precision)]
        after_merge = retrieval.merge_results_of_title_and_text_with_pageView_PageRank(list_new, page_views_pkl,
                                                                                       page_rank_pkl)

        top_100_merge = retrieval.get_top_n(after_merge)
        res = retrieval.result_doc_to_title(top_100_merge, docid_title_dict_pkl)

        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        logging.exception("error", e)
        respose = ('error', str(e))
        return jsonify(respose)


@app.route("/search_boolean_and_cosine")
def search_boolean_and_cosine():
    res = []
    query = request.args.get('query', '')
    try:
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        num_text_precision = 0.333
        num_title_precision = 0.667
        query_without_stem = retrieval.tokenize(query)
        if len(query_without_stem) <= 2:
            search_for_title = retrieval.boolean_search(query_without_stem, title_index)
            top_title = retrieval.get_top_n_in_dict(search_for_title)

            search_for_text = retrieval.cosine_similarity_search(query_without_stem, text_index,
                                                                 normalization_freq_dict_pkl)
            top_text = retrieval.get_top_n_in_dict(search_for_text)

            num_title_precision = 0.95
            num_text_precision = 0.05
        else:
            thread_search = retrieval.Search_with_thread()
            top_title, top_text = thread_search.search_thread(query_without_stem, query_without_stem, text_index,
                                                              title_index,
                                                              docid_count_dict_pkl, normalization_freq_dict_pkl)

        list_new = [(top_text, num_text_precision), (top_title, num_title_precision)]
        after_merge = retrieval.merge_results_of_title_and_text_with_pageView_PageRank(list_new, page_views_pkl,
                                                                                       page_rank_pkl)
        top_100_merge = retrieval.get_top_n(after_merge)
        res = retrieval.result_doc_to_title(top_100_merge, docid_title_dict_pkl)
        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        logging.exception("erorrrr", e)
        respose = ('error', str(e))
        return jsonify(respose)


@app.route("/search_bm25_and_cosine")
def search_bm25_and_cosine():
    res = []
    query = request.args.get('query', '')
    try:
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        num_text_precision = 0.677
        num_title_precision = 0.323
        query_without_stem = retrieval.tokenize(query)
        # query_without_stem = retrieval.tokenize_with_unique(query)

        if len(query_without_stem) <= 2:
            search_for_title = retrieval.BM25_search_text(query_without_stem, title_index, docid_count_dict_pkl, 1.5,
                                                          2.5)
            top_title = retrieval.get_top_n_in_dict(search_for_title)
            search_for_text = retrieval.cosine_similarity_search(query_without_stem, text_index,
                                                                 normalization_freq_dict_pkl)
            top_text = retrieval.get_top_n_in_dict(search_for_text)

            num_text_precision = 0.333
            num_title_precision = 0.667
        else:
            thread_search = retrieval.Search_with_thread()
            top_title, top_text = thread_search.search_thread(query_without_stem, query_without_stem, text_index,
                                                              title_index,
                                                              docid_count_dict_pkl, normalization_freq_dict_pkl)

        list_new = [(top_text, num_text_precision), (top_title, num_title_precision)]
        after_merge = retrieval.merge_results_of_title_and_text_with_pageView_PageRank(list_new, page_views_pkl,
                                                                                       page_rank_pkl)
        top_100_merge = retrieval.get_top_n(after_merge)
        res = retrieval.result_doc_to_title(top_100_merge, docid_title_dict_pkl)
        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        logging.exception("erorrrr", e)
        respose = ('error', str(e))
        return jsonify(respose)




@app.route("/search_body_BM25")
def search_body_BM25():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = retrieval.get_top_n(retrieval.BM25_search_text(query, text_index, docid_count_dict_pkl))
    res = retrieval.result_doc_to_title(res, docid_title_dict_pkl)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title_boolean")
def search_title_boolean():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = retrieval.get_top_n(retrieval.boolean_search(query, title_index))
    res = retrieval.result_doc_to_title(res, docid_title_dict_pkl)
    # END SOLUTION
    return jsonify(res)



"""It's the functions you provided that he didn't have to implement"""

# @app.route("/search_anchor")
# def search_anchor():
#     ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
#         IN THE ANCHOR TEXT of articles, ordered in descending order of the
#         NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
#         DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
#         3 (GCP part) to do the tokenization and remove stopwords. For example,
#         a document with a anchor text that matches two distinct query words will
#         be ranked before a document with anchor text that matches only one
#         distinct query word, regardless of the number of times the term appeared
#         in the anchor text (or query).
#
#         Test this by navigating to the a URL like:
#          http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
#         where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of ALL (not just top 100) search results, ordered from best to
#         worst where each element is a tuple (wiki_id, title).
#     '''
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#     # END SOLUTION
#     return jsonify(res)
#
#
# @app.route("/get_pagerank", methods=['POST'])
# def get_pagerank():
#     ''' Returns PageRank values for a list of provided wiki article IDs.
#
#         Test this by issuing a POST request to a URL like:
#           http://YOUR_SERVER_DOMAIN/get_pagerank
#         with a json payload of the list of article ids. In python do:
#           import requests
#           requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
#         As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of floats:
#           list of PageRank scores that correrspond to the provided article IDs.
#     '''
#     res = []
#     wiki_ids = request.get_json()
#     if len(wiki_ids) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)
#
#
# @app.route("/get_pageview", methods=['POST'])
# def get_pageview():
#     ''' Returns the number of page views that each of the provide wiki articles
#         had in August 2021.
#
#         Test this by issuing a POST request to a URL like:
#           http://YOUR_SERVER_DOMAIN/get_pageview
#         with a json payload of the list of article ids. In python do:
#           import requests
#           requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
#         As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of ints:
#           list of page view numbers from August 2021 that correrspond to the
#           provided list article IDs.
#     '''
#     res = []
#     wiki_ids = request.get_json()
#     if len(wiki_ids) == 0:
#         return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)


if __name__ == '__main__':
    init()
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    # app.run(host='0.0.0.0', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
