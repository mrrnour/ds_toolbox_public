####------------------------------Data Cleaning--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd


class TextSimilarity:
    """Compute similarity between two texts using a chosen method.

    Parameters
    ----------
    text1, text2 : str
        Strings to compare.
    method : str
        Name of the similarity method to dispatch to. One of
        ``'similarity_word2vec'``, ``'similarity_word_dist'``,
        ``'similarity_sbert'``.

    Attributes
    ----------
    text1 : str
        First text (stored).
    text2 : str
        Second text (stored).
    method : str
        Method name resolved via :meth:`calculate_similarity`.
    """

    def __init__(self, text1, text2, method):
        """Store the two texts and the dispatch method name."""
        self.text1 = text1
        self.text2 = text2
        self.method = method

    def similarity_word2vec(self):
        """Cosine similarity using spaCy's ``en_core_web_md`` word vectors.

        Returns
        -------
        float
            Similarity in ``[0, 1]`` from spaCy's ``Doc.similarity``.
        """
        import spacy
        nlp = spacy.load('en_core_web_md')  # load a model with word vectors
        doc1 = nlp(self.text1)
        doc2 = nlp(self.text2)
        return doc1.similarity(doc2)

    def similarity_word_dist(self):
        """Character-level edit-distance ratio via :class:`difflib.SequenceMatcher`.

        Returns
        -------
        float
            Ratio in ``[0, 1]``; 1.0 means identical.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, self.text1, self.text2).ratio()

    def similarity_sbert(self):
        """Cosine similarity of sentence embeddings (``all-MiniLM-L6-v2`` SBERT).

        Returns
        -------
        torch.Tensor
            1×1 tensor holding the cosine similarity.
        """
        import os
        os.environ['CURL_CA_BUNDLE'] = ''
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        #Compute embedding for both lists
        embedding_1= model.encode(self.text1, convert_to_tensor=False)
        embedding_2 = model.encode(self.text2, convert_to_tensor=False)

        return util.pytorch_cos_sim(embedding_1, embedding_2)

    def calculate_similarity(self):
        """Dispatch to the method named in ``self.method`` and return its score.

        Returns
        -------
        float or torch.Tensor
            Result of the dispatched method, or the string
            ``"Invalid method"`` if ``self.method`` is unknown.
        """
        method = getattr(self, self.method, lambda: "Invalid method")
        return method()

def calculate_similarity(df, column, similarity_method):
    """Compute pairwise similarity for every unique pair of values in a column.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    column : str
        Column whose unique values to compare pairwise.
    similarity_method : str
        Method name forwarded to :class:`TextSimilarity`.

    Returns
    -------
    pandas.DataFrame
        Columns ``<column>_1``, ``<column>_2``, ``similarity_score`` —
        one row per unique unordered pair.
    """
    from itertools import combinations
    unique_values = df[column].unique()
    similarity_scores = []
    for value1, value2 in combinations(unique_values, 2):
        similarity_score = TextSimilarity(value1, value2, similarity_method).calculate_similarity()
        similarity_scores.append([value1, value2, similarity_score])
    return pd.DataFrame(similarity_scores, columns=[column+'_1', column+'_2', 'similarity_score'])

def unify_similar_strings(df0, column, similarity_method='similarity_word_dist', threshold=.8):
    """Collapse near-duplicate strings in a column, keeping the shortest variant.

    Iterates through unique values sorted by length (shortest first); for any
    pair whose similarity meets ``threshold``, replaces the longer string with
    the shorter one across the whole column.

    Parameters
    ----------
    df0 : pandas.DataFrame
        Source frame; not modified (a copy is returned).
    column : str
        Column to deduplicate.
    similarity_method : str, optional
        Method forwarded to :class:`TextSimilarity`. Defaults to
        ``'similarity_word_dist'`` (fast edit-distance ratio).
    threshold : float, optional
        Minimum similarity to treat two strings as duplicates. Default 0.8.

    Returns
    -------
    tuple
        ``(df, replaced_df)`` where ``df`` is the cleaned copy and
        ``replaced_df`` lists every replacement made (columns
        ``Replaced``, ``Original``, ``Frequency``).
    """
    df=df0.copy()
    df[column] = df[column].str.strip().astype(str)
    unique_values =sorted(df[column].unique(), key=len, reverse=False)
    replaced_dict = {}
    i = 0
    while i < len(unique_values):
        j = i + 1
        while j < len(unique_values):
            similarity = TextSimilarity(unique_values[i], unique_values[j], similarity_method).calculate_similarity()
            if (similarity >= threshold) & (unique_values[i] != unique_values[j]):
                replaced_dict[unique_values[i]] = replaced_dict.get(unique_values[i], []) + [(unique_values[j], (df[column] == unique_values[j]).sum())]
                df.loc[df[column] == unique_values[j], column] = unique_values[i]
                unique_values = sorted(df[column].unique(), key=len, reverse=False)  # update unique_values
                break  # break the inner loop when we find a similar string
            j += 1
        i += 1

    replaced_df = pd.DataFrame([(k, v[0], v[1]) for k, values in replaced_dict.items() for v in values], columns=['Replaced', 'Original', 'Frequency'])

    return df , replaced_df

####------------------------------Lang Chain--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
