import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from rouge import Rouge
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import nltk
from rake_nltk import Rake

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Define functions for different summarization techniques
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Sentence segmentation
    sentences = nltk.sent_tokenize(text)
    return tokens, sentences

def summarize(text, summarizer_class, summary_length):
    tokens, sentences = preprocess_text(text)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = summarizer_class()
    summary = summarizer(parser.document, summary_length)
    return ' '.join([str(sentence) for sentence in summary]), sentences

def evaluate_summaries(reference, hypothesis):
    # Tokenize reference and hypothesis
    tokenized_reference = nltk.word_tokenize(reference)
    tokenized_hypothesis = nltk.word_tokenize(hypothesis)
    
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference)
    bleu_score = corpus_bleu([hypothesis], [[reference]]).score
    meteor_score = single_meteor_score(tokenized_reference, tokenized_hypothesis)
    
    return rouge_scores, bleu_score, meteor_score

def extract_keywords(text, num_keywords=10):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()
    return ranked_phrases[:num_keywords]

# Streamlit UI
st.set_page_config(page_title='SummarEase: Text Summarization App')
st.title('SummarEase: Customized summaries on Demand using ML')

# Developer credits
st.markdown("<p style='font-size: 14px;'>Developed by Chandana, Sana, Srija & Sumana</p>", unsafe_allow_html=True)

st.markdown("**Note:**")
st.markdown("🦋 Setting the summary length to 0.5 means you want a summary that's about half the length of the original text, giving you a condensed but still detailed overview.")
st.markdown("🦋 Choosing a summary length of 1.0 indicates that you want the summary to match the original text's length exactly, providing no summarization or condensing of information.")
st.markdown("🦋 Opting for a summary length of 0.1 means you're looking for an extremely brief summary, roughly one-tenth the length of the original text, capturing only the most essential points.")
st.markdown("\n")

user_input = st.text_area('Please enter your text here', height=200)
ratio = st.slider('Summary Length', min_value=0.1, max_value=1.0, step=0.1)
num_keywords = st.number_input('Number of Keywords', min_value=1, max_value=20, value=10)

if st.button('Get your customized summary :)'):
    if user_input.strip():
        summary_length = int(len(nltk.sent_tokenize(user_input)) * ratio)

        summary_lsa, sentences_lsa = summarize(user_input, LsaSummarizer, summary_length)
        summary_text_rank, sentences_text_rank = summarize(user_input, TextRankSummarizer, summary_length)
        summary_luhn, sentences_luhn = summarize(user_input, LuhnSummarizer, summary_length)

        # Reference summary for evaluation
        reference_summary = "The reference summary, which represents the gold standard summary for evaluation purposes."

        # Evaluate summaries using ROUGE, BLEU, and METEOR
        rouge_scores_lsa, bleu_score_lsa, meteor_score_lsa = evaluate_summaries(reference_summary, summary_lsa)
        rouge_scores_text_rank, bleu_score_text_rank, meteor_score_text_rank = evaluate_summaries(reference_summary, summary_text_rank)
        rouge_scores_luhn, bleu_score_luhn, meteor_score_luhn = evaluate_summaries(reference_summary, summary_luhn)

        # Extract keywords
        keywords = extract_keywords(user_input, num_keywords)

        # Display summaries, evaluation scores, and keywords
        st.markdown(f"**LSA Summary:** {summary_lsa}")
        st.markdown(f"**TextRank Summary:** {summary_text_rank}")
        st.markdown(f"**Luhn Summary:** {summary_luhn}")
        st.markdown("\n**Evaluation Scores:**")
        st.markdown(f"LSA - ROUGE Scores: {rouge_scores_lsa}")
        st.markdown(f"LSA - BLEU Score: {bleu_score_lsa}")
        st.markdown(f"LSA - METEOR Score: {meteor_score_lsa}")
        st.markdown(f"TextRank - ROUGE Scores: {rouge_scores_text_rank}")
        st.markdown(f"TextRank - BLEU Score: {bleu_score_text_rank}")
        st.markdown(f"TextRank - METEOR Score: {meteor_score_text_rank}")
        st.markdown(f"Luhn - ROUGE Scores: {rouge_scores_luhn}")
        st.markdown(f"Luhn - BLEU Score: {bleu_score_luhn}")
        st.markdown(f"Luhn - METEOR Score: {meteor_score_luhn}")
        
        st.markdown("\n**Extracted Keywords:**")
        st.markdown(", ".join(keywords))
    else:
        st.warning("Please enter some text to summarize!")
