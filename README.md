# Information Retrieval Final Project
# Introduction
This repository contains our final project for the Information Retrieval course at Ben-Gurion University, Israel. Our task was to develop a search engine based on the English Wikipedia corpus, enabling searches by title, body, or anchor text. The project encompasses building indexes for each main part of the document and implementing different retrieval methods to efficiently retrieve documents.

# Project Structure

**inverted_index_gcp.py:** Implements the creation of an inverted index, leveraging Google Cloud and the Spark library for index computation.

**retrieval.py:** Contains the implementation of the main retrieval methods used by our search engine, including Boolean model, Cosine Similarity, and BM25.

**search_frontend.py:** A Flask application that provides a RESTful API for interacting with the search engine, including endpoints for search and retrieving PageRank and pageview data.

*Setup and Installation*
Prerequisites
Python 3.x
Flask
NLTK
Google Cloud SDK
Apache Spark (for index computation)

# Running the Search Engine
Clone this repository to your local machine or server.
Install the required Python libraries using pip install -r requirements.txt.
Set up Google Cloud credentials by following these instructions.
To run the Flask application, navigate to the directory containing search_frontend.py and execute the command python search_frontend.py.
The search engine will be available at http://localhost:8080/search by default.
How We Retrieve Information
Indexing
We built the index of the corpus by indexing the body to the title and anchoring the text of a document. The indexing process utilizes Google Cloud and the Spark library for efficient computation.

# Main Retrieval Methods
Our search engine implements the following similarity methods:

- **Boolean Model:** Supports basic logical operations in queries.
- **Cosine Similarity:** Measures the cosine of the angle between vectors in a multi-dimensional space, providing relevance scores based on term frequency.
- **BM25:** An extension of the TF-IDF model, incorporating document length and term frequency saturation.
Dictionaries
The engine utilizes several dictionaries for managing term frequencies, document lengths, normalization factors, and title IDs.

# Our Best Search
Our best search implementation combines several sophisticated techniques to efficiently and effectively retrieve information from the English Wikipedia corpus. Here's an overview of our approach:

**1. Tokenization:** We start by tokenizing the query, which involves breaking down the query string into individual terms or tokens. This process is crucial for analyzing the query's components and matching them with the indexed documents.

**2. Dynamic Retrieval Strategy:**
- For short queries (up to 2 tokens), we employ a Boolean search for titles and a BM25 search for the text. This strategy is based on the assumption that short queries are likely to be more focused on specific titles or may require high precision in text matching.
- For longer queries, we engage a parallelized search using threading to concurrently search through text and title indices. This approach helps in handling more complex queries efficiently, leveraging both Boolean and BM25 models for a comprehensive search.
  
**3. Precision Weighting:** We assign different precision weights to text and title search results—0.677 for text and 0.323 for title searches by default. These weights were carefully chosen based on empirical results to balance the relevance of document titles and body text in the final search results. For short queries, we adjust the weights to 0.05 for text and 0.95 for title precision, emphasizing title matches.

**4. Merge Strategy:** After retrieving and weighting the search results, we merge them using a sophisticated merging strategy. This strategy takes into account PageRank and page views (using data from August 2021) for each document, providing a comprehensive relevance score that considers both the query-document match quality and the document's popularity or importance.

**5. Result Selection:** Finally, we select the top merged search results based on their cumulative scores. This selection ensures that users are presented with the most relevant and authoritative documents matching their query.

This search configuration represents our most effective and efficient approach to retrieving information from the English Wikipedia corpus. It showcases our ability to handle both short and long queries adaptively, ensuring high relevance and quality in the search results presented to the user.

# Conclusion
Developing an information retrieval engine posed both challenges and exciting opportunities. From building an index to fine-tuning retrieval methods for optimal results, this project required a mix of technical skills and attention to detail. Working with the vast corpus of English Wikipedia offered valuable insights and a rewarding experience.

# Contributors
- Lior Levy
- Lior Amitay

