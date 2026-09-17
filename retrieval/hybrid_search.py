from sklearn.metrics.pairwise import cosine_similarity
from .dense_vec import dense_vec
from .sparse_vec import sparse_vec

def hybrid_retrival(embedding_model, chunks, query, k=4):
    # DENSE SEARCH
    vectorstore = dense_vec(
        embedding_model,
        chunks
    )

    dense_doc = vectorstore.similarity_search(
        query,
        k=k
    )

    # TF-IDF SEARCH
    tfidf_vec, tfidf_matrix = sparse_vec(chunks)

    query_vector = tfidf_vec.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = scores.argsort()[-k:][::-1]

    tfidf_doc = [
        chunks[i]
        for i in top_indices
    ]

    # COMBINE RESULTS
    combined_doc = []

    for doc in dense_doc + tfidf_doc:
        if doc not in combined_doc:
            combined_doc.append(doc)


    return combined_doc[:k]