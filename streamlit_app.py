import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(x):
    return x

def preprocess(x):
    return x

@st.cache_resource
def load_models():
    scaler_rooms = joblib.load('scaler_rooms.pkl')
    scaler_price = joblib.load('scaler_price.pkl')
    scaler_area  = joblib.load('scaler_area.pkl')
    vectorizer   = joblib.load('vectorizer.pkl')
    backup       = pd.read_csv('backup.csv', index_col=0)

    # Recompute on startup
    address_vectors = vectorizer.transform(backup['adress']).toarray()
    rooms_vectors   = scaler_rooms.transform(backup[['beds', 'bath']])
    price_vectors   = scaler_price.transform(backup[['price']])
    area_vectors    = scaler_area.transform(backup[['area']])

    sim_rooms   = cosine_similarity(rooms_vectors)
    sim_price   = cosine_similarity(price_vectors)
    sim_area    = cosine_similarity(area_vectors)
    sim_address = cosine_similarity(address_vectors)

    final_similarity = (0.10 * sim_rooms +
                        0.10 * sim_price +
                        0.10 * sim_area  +
                        0.70 * sim_address)

    return scaler_rooms, scaler_price, scaler_area, vectorizer, final_similarity, backup

scaler_rooms, scaler_price, scaler_area, vectorizer, final_similarity, backup = load_models()


def fuzzy_address_match(address_list, tokens, threshold=80):
    for token in tokens:
        for item in address_list:
            if fuzz.ratio(token, item) >= threshold:
                return True
    return False


def hard_filter(beds=None, bath=None, price=None, area=None, address=None, fuzzy=False):
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
                lambda x: fuzzy_address_match(eval(x) if isinstance(x, str) else x, tokens)
            )]
        else:
            filtered = filtered[filtered['adress'].apply(
                lambda x: any(token in (eval(x) if isinstance(x, str) else x) for token in tokens)
            )]
    return filtered


def parse_query(query):
    query = query.lower()
    beds  = re.search(r'(\d+)\s*bed', query)
    bath  = re.search(r'(\d+)\s*bath', query)
    area  = re.search(r'(\d+)\s*(?:sqft|sft|sq)', query)
    beds  = int(beds.group(1)) if beds else None
    bath  = int(bath.group(1)) if bath else None
    area  = int(area.group(1)) if area else None
    address = re.sub(r'\d+\s*(?:bed|bath|tk|taka|bdt|sqft|sft|sq)?', '', query).strip()
    address = address if address else None
    numbers = re.findall(r'\b(\d+)\b', query)
    used = [str(beds), str(bath), str(area)]
    price_candidates = [int(n) for n in numbers if n not in used]
    price = max(price_candidates) if price_candidates else None
    return beds, bath, price, area, address


def query_to_similarity(beds=None, bath=None, price=None, area=None, address=None):
    sims = []
    weights = []
    if beds is not None or bath is not None:
        b  = beds if beds is not None else backup['beds'].mean()
        bt = bath if bath is not None else backup['bath'].mean()
        input_rooms = scaler_rooms.transform([[b, bt]])
        sims.append(cosine_similarity(input_rooms, scaler_rooms.transform(backup[['beds', 'bath']]))[0])
        weights.append(1)
    if price is not None:
        input_price = scaler_price.transform([[price]])
        sims.append(cosine_similarity(input_price, scaler_price.transform(backup[['price']]))[0])
        weights.append(1)
    if area is not None:
        input_area = scaler_area.transform([[area]])
        sims.append(cosine_similarity(input_area, scaler_area.transform(backup[['area']]))[0])
        weights.append(1)
    if address is not None:
        input_address = vectorizer.transform([address.lower().split()]).toarray()
        sims.append(cosine_similarity(input_address, vectorizer.transform(backup['adress']).toarray())[0])
        weights.append(1)
    total = sum(weights)
    final = sum((w / total) * s for w, s in zip(weights, sims))
    return final


def fetch_and_recommend(query, top_n=5):
    beds, bath, price, area, address = parse_query(query)

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
    if filtered.empty:
        scores = query_to_similarity(beds, bath, price, area, address)
        top_indices = scores.argsort()[::-1][:top_n]
        result = backup.iloc[top_indices].copy()
        result['similarity'] = scores[top_indices]
        return None, result

    best_index = filtered.index[0]
    best_match = backup.loc[best_index]

    scores = list(enumerate(final_similarity[best_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != best_index]

    seen_addresses = set()
    diverse_indices = []
    for i, s in scores:
        addr = str(backup.iloc[i]['adress'])
        if addr not in seen_addresses:
            seen_addresses.add(addr)
            diverse_indices.append((i, s))
        if len(diverse_indices) == top_n:
            break

    top_indices = [i for i, _ in diverse_indices]
    result = backup.iloc[top_indices].copy()
    result['similarity'] = [s for _, s in diverse_indices]
    return best_match, result


# Streamlit UI
st.title("🏠 Property Recommender")
st.write("Search for a property and get similar recommendations")

query = st.text_input("Search", placeholder="e.g. 2 bed gulshan 50000")

if query:
    with st.spinner("Finding properties..."):
        best_match, results = fetch_and_recommend(query)

    if results is None:
        st.error("No properties found. Try a different search.")
    else:
        if best_match is not None:
            st.subheader("Best Match")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Beds", best_match['beds'])
            col2.metric("Baths", best_match['bath'])
            col3.metric("Price", f"{best_match['price']:,.0f}")
            col4.metric("Area", f"{best_match['area']:,.0f} sqft")
            st.write(f"📍 {best_match['adress']}")
            st.write(f"**{best_match['title']}**")

        st.subheader("Similar Properties")
        for _, row in results.iterrows():
            with st.expander(f"{row['title']}  |  similarity: {row['similarity']:.2f}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Beds", row['beds'])
                c2.metric("Baths", row['bath'])
                c3.metric("Price", f"{row['price']:,.0f}")
                c4.metric("Area", f"{row['area']:,.0f} sqft")
                st.write(f"📍 {row['adress']}")
