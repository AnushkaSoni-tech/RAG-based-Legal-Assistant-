from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def sparse_vec(chunks):
    tfidf_vec=TfidfVectorizer()
    tfidf_matrix=tfidf_vec.fit_transform([chunk.page_content for chunk in chunks])
    return tfidf_vec,tfidf_matrix