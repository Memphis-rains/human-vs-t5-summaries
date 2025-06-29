# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
import textstat

st.set_page_config(page_title="Human vs T5 Summary Analysis", layout="wide")
st.markdown("<style>.main{background:white;color:black;}</style>", unsafe_allow_html=True)
sns.set_style("whitegrid")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load Data and Model (cached)
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data('data/sampled_100_with_t5_summary.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute features (cached)

def compute_features(df):
    def feats(txt):
        w, s = word_tokenize(txt), sent_tokenize(txt)
        return {
            'word_count': len(w),
            'sentence_count': len(s),
            'char_count': len(txt),
            'punct_count': sum(c in string.punctuation for c in txt),
            'sentence_density': len(s)/(len(w)+1e-6),
            'word_density': len(w)/(len(txt)+1e-6),
            'readability': textstat.flesch_kincaid_grade(txt)
        }
    human = df['summary'].apply(feats).apply(pd.Series).add_suffix('_human')
    t5 = df['t5_summary'].apply(feats).apply(pd.Series).add_suffix('_t5')
    feats_df = pd.concat([human, t5], axis=1)
    feats_df['compression_ratio_human'] = df['summary'].str.len() / df['text'].str.len()
    feats_df['compression_ratio_t5'] = df['t5_summary'].str.len() / df['text'].str.len()
    return feats_df

feats = compute_features(df)


def compute_metrics(df, feats, model):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge = df.apply(lambda r: scorer.score(r['summary'], r['t5_summary']), axis=1)
    bleu = df.apply(lambda r: sentence_bleu([word_tokenize(r['summary'].lower())], word_tokenize(r['t5_summary'].lower())), axis=1)
    emb_h = model.encode(df['summary'].tolist(), convert_to_tensor=True)
    emb_t = model.encode(df['t5_summary'].tolist(), convert_to_tensor=True)
    cosine = util.cos_sim(emb_h, emb_t).diagonal().cpu().numpy()
    metrics_df = pd.DataFrame({
        'ROUGE-1':[v['rouge1'].fmeasure for v in rouge],
        'ROUGE-2':[v['rouge2'].fmeasure for v in rouge],
        'ROUGE-L':[v['rougeL'].fmeasure for v in rouge],
        'BLEU':bleu,
        'Cosine Similarity':cosine,
        'Readability_human':feats['readability_human'],
        'Readability_t5':feats['readability_t5'],
        'Compression_human':feats['compression_ratio_human'],
        'Compression_t5':feats['compression_ratio_t5'],
    })
    return metrics_df

metrics_df = compute_metrics(df, feats, model)

st.title("📊 Human vs T5 Summary Analysis")

# Dataset Overview
st.header("Dataset Overview")

st.markdown("""
**Original Dataset:**  
[CNN/DailyMail on HuggingFace](https://huggingface.co/datasets/abisee/cnn_dailymail)

**Download Links:**    
- [Download sample dataset](https://your-link-to-sample-dataset.com)  

**Description:**  
The CNN/DailyMail dataset consists of news articles and their human-written summaries.  
It is widely used for training and evaluating abstractive text summarization models.  
Each sample contains an article and a multi-sentence summary (highlight), making it ideal for benchmarking models like T5.  
The dataset is large, diverse, and representative of real-world news summarization tasks.
""")

st.write(f"Total records: **{len(df)}**")
st.dataframe(df[['summary', 't5_summary']].head(), use_container_width=True)

# Custom observation texts
obs_text = {
    'word_count':     """

1) T5 summaries average 50.8 words versus 52.0 for humans which is nearly identical.

2) Both distributions peak around 40–60 words, showing T5 matches typical human summary lengths.

3) Humans occasionally write much longer summaries; T5 outputs stay more tightly clustered.

4) T5’s slight brevity and uniformity can improve predictability without sacrificing human‐like length.""",



    'sentence_count':   """

     1) Human summaries span about 3.8 sentences on average, whereas T5 settles at roughly 2.9 sentences—a clear tendency toward tighter, more concise outputs.

     2) The T5 density curve peaks sharply around 3 sentences, indicating it almost always uses a similar sentence count; by contrast, human summaries spread broadly from 2–5 sentences.

     3) Humans occasionally write very short or longer multi-sentence summaries, but T5 keeps its outputs within a narrow band—great for consistency but potentially at the expense of nuanced detail.

     4) While T5’s uniform sentence structure boosts predictability, human summaries’ greater sentence-count diversity may capture extra context that a fixed-length model could miss.
    """,


    'char_count':     """

   1) On average, T5 produces summaries of ~274.9 characters versus 286.1 for human authors—a modest 4% reduction in length.
   
   2) Both density curves peak around 220–280 characters, showing T5 reliably captures the core human summary length.
   
   3) Human character counts extend into much longer summaries (500+ characters), whereas T5 remains tightly clustered—trading occasional extra detail for consistency.

   4) T5’s narrower spread suggests predictable output lengths, which can be advantageous in applications requiring uniform summary sizes without dramatically deviating from human norms.
    
    """,
    'punct_count':    """
    
    1) Human summaries use about 5.3 punctuation marks on average versus 5.2 for T5—practically the same.
    
    2) Both density curves peak around 4–6 punctuation marks, showing T5 reliably mirrors human punctuation frequency.
    
    3) T5’s density is slightly higher in the 5–7 range, indicating it favors a moderate punctuation level.
    
    4) Humans exhibit a longer tail into higher punctuation counts (10+), suggesting occasional more elaborate or emphatic phrasing that T5 tends to avoid.
    """,
    
}


# Comparison Metrics
st.header("Feature Comparisons")

features = [
    ('word_count', 'Word Count'),
    ('sentence_count', 'Sentence Count'),
    ('char_count', 'Character Count'),
    ('punct_count', 'Punctuation Count'),
    
]
for key, name in features:
    st.subheader(f"{name} Comparison")

    # Two columns: count, density
    c1, c2 = st.columns([3, 3], gap="large")

    # Bar chart (Count)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        vals = [feats[f"{key}_human"].mean(), feats[f"{key}_t5"].mean()]
        sns.barplot(x=['Human', 'T5'], y=vals, palette="Set2", ax=ax1)
        for p in ax1.patches:
            ax1.annotate(f"{p.get_height():.1f}", 
                         (p.get_x()+p.get_width()/2, p.get_height()),
                         ha='center', va='bottom', fontsize=12)
        ax1.set_facecolor('white')
        ax1.set_ylabel(name)
        ax1.set_xlabel("")
        st.pyplot(fig1)

    # Density plot
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.kdeplot(feats[f"{key}_human"], fill=True, label='Human', ax=ax2, color="skyblue")
        sns.kdeplot(feats[f"{key}_t5"], fill=True, label='T5', ax=ax2, color="salmon")
        ax2.legend()
        ax2.set_facecolor('white')
        ax2.set_ylabel("Density")
        ax2.set_xlabel(name)
        st.pyplot(fig2)

    # Observations tab (full width)
    st.markdown(f"**Observations:** {obs_text[key]}") 

# Evaluation Metrics Summary
st.header("Evaluation Metrics Summary")
summary_table = metrics_df.mean().to_frame("Mean Score").round(3)
st.table(summary_table)

# Readability and Linguistic Quality
st.header("Readability and Linguistic Quality Metrics")
st.markdown("""
- **Compression Ratio**: Measures summary conciseness  
  $$\\frac{\\text{Length of summary}}{\\text{Length of original text}}$$
- Lower values indicate higher conciseness.
""")

# Overall Conclusion (Customizable)
st.header("Overall Conclusion")
This project uses a 100-sample subset from the well-known CNN/DailyMail dataset to compare human-written and T5-generated summaries in detail. We assess both summarization styles using a variety of quantitative and qualitative metrics.

The findings demonstrate a striking similarity in the fundamental structure and length of summaries produced by AI and humans. Both distributions peak in the 40–60 word range, and the average word count for T5 summaries (50.8) closely resembles that of human-written summaries (52.0), demonstrating that T5 effectively emulates human summarization patterns. The model's innate propensity for uniformity and brevity is demonstrated by the fact that T5 outputs stay more closely clustered while human summaries can occasionally reach much longer lengths.
         
A similar pattern emerges for sentence and character counts. T5 summaries are generally shorter (about 2.9 sentences, 274.9 characters) compared to their human counterparts (3.8 sentences, 286.1 characters). This reflects T5’s bias for concise outputs and greater consistency. Human summaries, by contrast, demonstrate more variation, sometimes offering shorter or much longer responses—potentially providing additional nuance or context.

Punctuation analysis further demonstrates that T5 closely mirrors human behavior, with both averaging around five punctuation marks per summary. However, human authors occasionally use much more punctuation, hinting at more complex or expressive sentence structures.

When considering evaluation metrics, T5-generated summaries achieve a ROUGE-1 score of 0.394, ROUGE-L of 0.281, and cosine similarity of 0.71, indicating a strong semantic and lexical resemblance to human summaries. Readability scores show both are accessible, though T5 summaries may be slightly simpler on average.

Overall, T5 proves highly capable at replicating the structure, length, and general readability of human-written summaries. While it tends to be more consistent and concise, some richness and variability unique to human writing may be diminished. These findings underscore the impressive progress of modern text summarization models, while also highlighting the subtle distinctions that still separate AI outputs from human creativity and flexibility.
""")

