import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Must be defined before loading any pickled objects ──────────
def tokenize(x):
    return x

def preprocess(x):
    return x


# ─── Load Everything ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler_rooms     = joblib.load('scaler_rooms.pkl')
    scaler_price     = joblib.load('scaler_price.pkl')
    scaler_area      = joblib.load('scaler_area.pkl')
    address_vectors  = np.load('address_vectors.npy')
    title_vectors    = np.load('title_vectors.npy')
    weighted_vectors = np.load('weighted_vectors.npy').astype('float32')
    index            = faiss.read_index('property_index.faiss')
    backup           = pd.read_csv('backup.csv', index_col=0)
    sentence_model   = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert adress back to list
    backup['adress'] = backup['adress'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    # Precompute average vectors for intent detection
    avg_address_vec = address_vectors.mean(axis=0).reshape(1, -1)
    avg_title_vec   = title_vectors.mean(axis=0).reshape(1, -1)

    return (scaler_rooms, scaler_price, scaler_area,
            address_vectors, title_vectors, weighted_vectors,
            index, backup, sentence_model,
            avg_address_vec, avg_title_vec)


(scaler_rooms, scaler_price, scaler_area,
 address_vectors, title_vectors, weighted_vectors,
 index, backup, sentence_model,
 avg_address_vec, avg_title_vec) = load_models()

dimension = weighted_vectors.shape[1]


# ─── Constants ───────────────────────────────────────────────────
STOPWORDS = {
    'low', 'high', 'cheap', 'affordable', 'luxury', 'priced', 'price',
    'property', 'apartment', 'flat', 'house', 'in', 'at', 'near',
    'with', 'and', 'the', 'a', 'an', 'for', 'rent', 'sale', 'available',
    'looking', 'want', 'need', 'find', 'me', 'i', 'is', 'are', 'bedroom',
    'bathroom', 'room', 'rooms', 'bed', 'bath'
}


# ─── Helper Functions ─────────────────────────────────────────────
def words_to_numbers(query):
    numbers = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    for word, num in numbers.items():
        query = re.sub(rf'\b{word}\b', num, query)
    return query


def fuzzy_address_match(address_list, tokens, threshold=80):
    for token in tokens:
        for item in address_list:
            if fuzz.ratio(token, item) >= threshold:
                return True
    return False


def detect_query_intent(query):
    query_vec     = sentence_model.encode(query).reshape(1, -1)
    address_score = cosine_similarity(query_vec, avg_address_vec)[0][0]
    title_score   = cosine_similarity(query_vec, avg_title_vec)[0][0]
    total         = address_score + title_score
    w_address     = address_score / total
    w_title       = title_score   / total
    return w_address, w_title


def hard_filter(beds=None, bath=None, price=None, area=None,
                address=None, fuzzy=False):
    filtered = backup.copy()
    if beds is not None:
        filtered = filtered[filtered['beds'] == beds]
    if bath is not None:
        filtered = filtered[filtered['bath'] == bath]
    if price is not None:
        filtered = filtered[filtered['price'].between(price * 0.8, price * 1.2)]
    if area is not None:
        filtered = filtered[filtered['area'].between(area * 0.8, area * 1.2)]
    if address is not None:
        tokens = address.lower().split()
        if fuzzy:
            filtered = filtered[filtered['adress'].apply(
                lambda x: fuzzy_address_match(x, tokens)
            )]
        else:
            filtered = filtered[filtered['adress'].apply(
                lambda x: any(token in x for token in tokens)
            )]
    return filtered


def parse_query(query):
    query = query.lower()
    query = words_to_numbers(query)

    lowest_price  = any(w in query for w in
                       ['low price', 'lowest price', 'cheap',
                        'affordable', 'budget', 'low priced'])
    highest_price = any(w in query for w in
                       ['luxury', 'expensive', 'premium', 'high end'])

    beds = re.search(r'(\d+)\s*bed', query)
    bath = re.search(r'(\d+)\s*bath', query)
    area = re.search(r'(\d+)\s*(?:sqft|sft|sq)', query)
    beds = int(beds.group(1)) if beds else None
    bath = int(bath.group(1)) if bath else None
    area = int(area.group(1)) if area else None

    numbers          = re.findall(r'\b(\d+)\b', query)
    used             = [str(beds), str(bath), str(area)]
    price_candidates = [int(n) for n in numbers if n not in used]
    price            = max(price_candidates) if price_candidates else None

    address = re.sub(r'\d+\s*(?:bed|bath|tk|taka|bdt|sqft|sft|sq)?', '', query)
    for word in STOPWORDS:
        address = re.sub(rf'\b{word}\b', '', address)
    address = re.sub(r'\s+', ' ', address).strip()
    address = address if address else None

    return beds, bath, price, area, address, lowest_price, highest_price


def get_query_vector(beds=None, bath=None, price=None, area=None,
                     address=None, query=None):
    w_address, w_title = detect_query_intent(query) if query else (0.50, 0.30)

    b  = beds  if beds  is not None else backup['beds'].mean()
    bt = bath  if bath  is not None else backup['bath'].mean()
    p  = price if price is not None else backup['price'].mean()
    ar = area  if area  is not None else backup['area'].mean()
    ad = address if address is not None else query if query else ''
    kw = query   if query   is not None else ''

    rooms_vec   = scaler_rooms.transform([[b, bt]])
    price_vec   = scaler_price.transform([[p]])
    area_vec    = scaler_area.transform([[ar]])
    address_vec = sentence_model.encode(ad).reshape(1, -1)
    title_vec   = sentence_model.encode(kw).reshape(1, -1)

    combined = np.hstack([
        0.07 * rooms_vec,
        0.07 * price_vec,
        0.06 * area_vec,
        w_address * address_vec,
        w_title   * title_vec
    ]).astype('float32')

    faiss.normalize_L2(combined)
    return combined


def fetch_and_recommend(query, top_n=5):
    beds, bath, price, area, address, lowest_price, highest_price = parse_query(query)

    w_address, w_title = detect_query_intent(query)

    filtered = hard_filter(beds, bath, price, area, address)
    if filtered.empty:
        filtered = hard_filter(beds, bath, address=address)
    if filtered.empty:
        filtered = hard_filter(beds, bath, price, area, address, fuzzy=True)
    if filtered.empty:
        filtered = hard_filter(beds, bath, address=address, fuzzy=True)
    if filtered.empty:
        filtered = hard_filter(beds=beds, address=address, fuzzy=True)
    if filtered.empty:
        filtered = hard_filter(bath=bath, address=address, fuzzy=True)
    if filtered.empty:
        filtered = hard_filter(price=price, address=address, fuzzy=True)
    if filtered.empty:
        filtered = hard_filter(address=address, fuzzy=True)
    if filtered.empty:
        filtered = hard_filter(beds=beds, bath=bath)
    if filtered.empty:
        filtered = hard_filter(beds=beds)
    if filtered.empty:
        filtered = hard_filter(bath=bath)

    if not filtered.empty:
        query_title_vec      = sentence_model.encode(query).reshape(1, -1)
        candidate_title_vecs = title_vectors[filtered.index.tolist()]
        title_scores         = cosine_similarity(query_title_vec, candidate_title_vecs)[0]

        filtered             = filtered.copy()
        filtered['similarity'] = title_scores

        if w_address > w_title:
            filtered = filtered.sort_values('similarity', ascending=False)
        else:
            all_title_scores = cosine_similarity(query_title_vec, title_vectors)[0]
            filtered         = backup.copy()
            filtered['similarity'] = all_title_scores
            if beds is not None:
                filtered = filtered[filtered['beds'] == beds]
            if bath is not None:
                filtered = filtered[filtered['bath'] == bath]
            if price is not None:
                filtered = filtered[filtered['price'].between(price * 0.8, price * 1.2)]
            filtered = filtered.sort_values('similarity', ascending=False)

        best_index = filtered.index[0]
        best_match = backup.loc[best_index]
        result     = filtered.iloc[1:top_n+1].copy()

        if lowest_price:
            result = result.sort_values('price', ascending=True)
        elif highest_price:
            result = result.sort_values('price', ascending=False)

        return best_match, result

    # FAISS fallback
    query_vec            = get_query_vector(beds, bath, price, area, address, query)
    distances, indices   = index.search(query_vec, top_n)
    result               = backup.iloc[indices[0]].copy()
    result['similarity'] = distances[0]
    return None, result


# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(page_title="Property Recommender", page_icon="🏠", layout="wide")
st.title("🏠 Property Recommender")
st.write("Search by area, beds, bath, price or any combination")

query = st.text_input("Search", placeholder="e.g. 2 bed gulshan 50000")

if query:
    with st.spinner("Finding properties..."):
        best_match, results = fetch_and_recommend(query)

    if results is None or len(results) == 0:
        st.error("No properties found. Try a different search.")
    else:
        if best_match is not None:
            st.subheader("🎯 Best Match")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Beds",  int(best_match['beds']))
            col2.metric("Baths", int(best_match['bath']))
            col3.metric("Price", f"{best_match['price']:,.0f}")
            col4.metric("Area",  f"{best_match['area']:,.0f} sqft")
            st.write(f"📍 {best_match['adress']}")
            st.write(f"**{best_match['title']}**")
            st.divider()

        st.subheader("🏘️ Similar Properties")
        for _, row in results.iterrows():
            with st.expander(f"🏠 {row['title']}  |  similarity: {row['similarity']:.2f}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Beds",  int(row['beds']))
                c2.metric("Baths", int(row['bath']))
                c3.metric("Price", f"{row['price']:,.0f}")
                c4.metric("Area",  f"{row['area']:,.0f} sqft")
                st.write(f"📍 {row['adress']}")
```

Push all files to GitHub:
```
app.py
requirements.txt
scaler_rooms.pkl
scaler_price.pkl
scaler_area.pkl
address_vectors.npy
title_vectors.npy
weighted_vectors.npy
property_index.faiss
backup.csv
