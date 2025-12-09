"""
arabic_sentiment_pipeline.py
---------------------------------

This module implements a robust, modular pipeline for performing sentiment
analysis on Arabic product reviews.  The design follows a three‑module
architecture – pre‑processing, sentiment classification and topic extraction –
which can be composed into a single pipeline.  Each module is carefully
documented and relies on well‑established libraries where possible.  Users
should install the required dependencies (`camel_tools`, `transformers`,
`torch`, `yake`, etc.) before running the pipeline.

Key references consulted during design:

* **Arb‑MCNN‑Bi model for sentiment analysis** – a hybrid architecture that
  combines AraBERT embeddings with multi‑channel convolutional and
  bidirectional GRU layers, achieving high accuracy on multiple Arabic
  review datasets【983578814277876†L520-L533】.  This forms the basis of the
  sentiment classification module.
* **CAMeL Tools** – an open source toolkit providing utilities for
  pre‑processing, morphological modelling, dialect identification and
  sentiment analysis【335548642306055†L9-L14】.  We leverage its
  normalisation, morphological tokenisation and dialect identification
  capabilities wherever available.
* **Dialect Identification** – CAMeL’s dialect identifier can distinguish
  between 25 Arabic city dialects and Modern Standard Arabic【920536881325592†L45-L50】.
  Dialect awareness can optionally guide downstream processing.
* **Morphological Tokenisation** – CAMeL’s morphological tokenizer offers
  fine‑grained tokenisation based on morphological analysis, configurable
  through different schemes【473074898223541†L34-L78】.  When such
  resources are unavailable, simpler word tokenisation is used as a
  fallback.

The goal is to provide a professional, production‑ready pipeline that
addresses the challenges of Arabic text processing – rich morphology,
dialectal variation and complex sentiment cues – while remaining flexible
enough for further fine‑tuning or extension.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

# Optional third‑party imports.  These are declared inside functions
# rather than at module scope so that the module can still be imported
# when optional dependencies are missing.  Users are responsible for
# installing the required packages prior to running the pipeline.


class ArabicPreprocessor:
    """Perform thorough pre‑processing on Arabic text.

    This class normalises Arabic text, removes extraneous characters,
    tokenises into words or morphological units, and optionally performs
    dialect identification.  It uses the CAMeL Tools library when
    available, but falls back to simpler heuristics when necessary.

    The normalisation step is deliberately conservative: it removes
    diacritics, tatweel characters, unifies hamza variants and
    alef/yaa variants, strips punctuation and collapses whitespace.

    Attributes
    ----------
    dialect_identifier : Optional[object]
        An instance of ``camel_tools.dialectid.DialectIdentifier`` for
        dialect detection.  If ``None``, dialect identification is not
        performed.
    logger : logging.Logger
        Logger used for diagnostics.
    """

    def __init__(self, enable_dialect_id: bool = False) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dialect_identifier = None
        if enable_dialect_id:
            try:
                # Deferred import – only import if enabled
                from camel_tools.dialectid import DialectIdentifier
                self.dialect_identifier = DialectIdentifier.pretrained()
                self.logger.info(
                    "Loaded CAMeL dialect identifier (Model‑26) for 25 city dialects"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load CAMeL dialect identifier – dialect identification disabled: {e}"
                )

    @staticmethod
    def _basic_normalize(text: str) -> str:
        """Apply lightweight Arabic normalisation.

        Removes diacritics, tatweel, unifies Alef/hamza variants and
        Yaa/Alef maqsuura.  Also strips non‑alphanumeric punctuation and
        collapses whitespace.  This function does not depend on external
        libraries and serves as a fallback when CAMeL normalisation is
        unavailable.
        """
        import re

        # Arabic diacritics and tatweel
        arabic_diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
        tatweel_char = "\u0640"
        text = arabic_diacritics.sub("", text)
        text = text.replace(tatweel_char, "")
        # Unify hamza/alef variants
        text = re.sub("[\u0622\u0623\u0625]", "ا", text)
        # Normalise yaa/alef maqsuura
        text = text.replace("ى", "ي")
        # Replace non‑alphanumeric punctuation with space (keep Arabic letters/digits)
        text = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize(self, text: str) -> str:
        """Perform normalisation using CAMeL Tools if available, else fallback.

        This method attempts to use ``camel_tools.utils.normalize`` if
        installed.  When unavailable, it falls back to :meth:`_basic_normalize`.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Normalised text.
        """
        # Try to use CAMeL Tools normaliser
        try:
            from camel_tools.utils.normalize import normalize_arabic
            norm = normalize_arabic(text)
            # Additional cleaning: collapse whitespace and strip
            import re
            norm = re.sub(r"\s+", " ", norm).strip()
            return norm
        except Exception:
            # Fallback to basic normalisation
            return self._basic_normalize(text)

    def tokenize(self, text: str, scheme: str = "simple") -> List[str]:
        """Tokenise Arabic text into a list of tokens.

        When CAMeL Tools is available, morphological tokenisation is used
        (via ``camel_tools.tokenizers.word`` or ``camel_tools.tokenizers.morphological``).
        The ``scheme`` parameter can be ``'simple'`` for whitespace/\
        punctuation tokenisation or any tokenisation scheme supported by
        ``MorphologicalTokenizer`` (e.g. ``'bwtok'``, ``'atbtok'``) if the
        necessary resources are available.  If CAMeL Tools is not
        installed, a simple whitespace split is used.

        Parameters
        ----------
        text : str
            Input text (assumed to be normalised already).
        scheme : str, default "simple"
            Tokenisation scheme.  ``'simple'`` uses basic word tokenisation;
            other values require CAMeL Tools and must be a valid scheme
            returned by ``tok_feats()`` of the chosen disambiguator.

        Returns
        -------
        List[str]
            List of token strings.
        """
        # If scheme is simple, attempt to use CAMeL's simple tokenizer
        if scheme == "simple":
            try:
                from camel_tools.tokenizers.word import simple_word_tokenize
                return simple_word_tokenize(text)
            except Exception:
                # Fallback: naive whitespace split
                return text.split()
        # Otherwise attempt morphological tokenisation
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            from camel_tools.tokenizers.morphological import MorphologicalTokenizer
            # Use MSA disambiguator by default
            disambig = MLEDisambiguator.pretrained("calima-msa-r13")
            morph_tokenizer = MorphologicalTokenizer(
                disambiguator=disambig, scheme=scheme, split=True
            )
            words = text.split()
            # Tokenise each word and flatten
            tokens: List[str] = []
            for tok in morph_tokenizer.tokenize(words):
                tokens.extend(tok)
            return tokens
        except Exception as e:
            self.logger.warning(
                f"Morphological tokenisation unavailable (scheme={scheme}); falling back to simple tokenisation: {e}"
            )
            return text.split()

    def detect_dialect(self, text: str) -> Optional[str]:
        """Detect the dialect of the given text.

        Returns a short label such as 'EGY', 'MSA', or the name of a city
        dialect.  Requires that a dialect identifier be initialised on
        construction.  If detection fails or no dialect identifier is
        available, ``None`` is returned.
        """
        if not self.dialect_identifier:
            return None
        try:
            result = self.dialect_identifier.predict([text], output="country")
            return result[0].top if result else None
        except Exception as e:
            self.logger.warning(f"Dialect detection failed: {e}")
            return None


class ArbMCNNBiClassifier:
    """AraBERT + Multi‑Channel CNN + BiGRU sentiment classifier.

    This model implements the hybrid architecture proposed by
    Almaqtari et al. (2024) in their ``Arb‑MCNN‑Bi`` model, which
    combines contextual embeddings from a pre‑trained AraBERT model
    with parallel convolutional filters and a bidirectional GRU
    classifier.  The architecture is summarised as follows:

    1. **Base Encoder** – a frozen AraBERT model from Hugging Face
       (e.g. ``aubmindlab/bert-base-arabertv02``).  Only its
       embeddings are used.
    2. **Convolutional Layers** – three parallel 1D convolution
       layers with kernel sizes of 3, 4 and 5 capturing local
       n‑gram features.  Each convolution is followed by ReLU
       activation and max‑over‑time pooling.
    3. **BiGRU Layer** – a bidirectional GRU processes the
       concatenated convolutional feature maps to capture global
       dependencies.
    4. **Classification Layer** – a feedforward layer outputs
       sentiment scores (positive/negative/neutral by default).

    The model is designed for fine‑tuning on downstream sentiment
    datasets.  At initialisation, the BERT parameters can be frozen
    or left trainable via the ``freeze_bert`` flag.  After training,
    call ``predict`` to obtain sentiment logits and probabilities.
    """

    def __init__(
        self,
        bert_model_name: str = "aubmindlab/bert-base-arabertv02",
        num_labels: int = 3,
        freeze_bert: bool = True,
    ) -> None:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.logger = logging.getLogger(self.__class__.__name__)
        # Load pre‑trained AraBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.num_labels = num_labels
        # Channel sizes for parallel convolutions
        conv_out_channels = 128
        kernel_sizes = [3, 4, 5]
        hidden_size = 128
        # Convolution layers
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=self.bert.config.hidden_size,
                          out_channels=conv_out_channels,
                          kernel_size=k)
                for k in kernel_sizes
            ]
        )
        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=conv_out_channels * len(kernel_sizes),
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        # Fully connected classification layer
        self.fc = nn.Linear(hidden_size * 2, num_labels)
        # Softmax for probabilities
        self.softmax = nn.Softmax(dim=1)

    def _encode(self, text: str, max_length: int = 128) -> Tuple[Any, Any]:
        """Tokenise and encode a single text for the BERT model.

        Returns input IDs and attention mask tensors of shape (1, L).
        """
        import torch

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def forward(self, input_ids: Any, attention_mask: Any) -> Any:
        """Compute logits given encoded inputs.

        This method should be called within a torch.no_grad() or training
        context and accepts batched input IDs and attention masks.
        """
        import torch
        import torch.nn as nn

        # BERT encoding: last hidden state of shape (B, L, H)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, L, H)
        # Swap to (B, H, L) for Conv1d
        x = last_hidden_state.permute(0, 2, 1)
        # Apply convolutions + ReLU + max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_x = torch.relu(conv(x))  # (B, C, L')
            # Max over time
            pooled = torch.max(conv_x, dim=2).values  # (B, C)
            conv_outputs.append(pooled)
        # Concatenate features from all kernels
        concat = torch.cat(conv_outputs, dim=1).unsqueeze(1)  # (B, 1, C*k)
        # BiGRU expects (B, seq_len, input_size); here seq_len=1
        gru_output, _ = self.bigru(concat)
        # Output is (B, seq_len, hidden_size*2); take the first timestep
        gru_feature = gru_output[:, 0, :]
        logits = self.fc(gru_feature)
        return logits

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment scores for a single text string.

        Returns a dictionary containing both raw logits and a probability
        distribution over sentiment labels.  The label mapping is assumed to
        be ``{0: 'negative', 1: 'neutral', 2: 'positive'}``, but users can
        interpret the indices as needed.
        """
        import torch
        # Switch to evaluation mode
        self.bert.eval()
        # Encode text
        input_ids, attention_mask = self._encode(text)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = self.softmax(logits)[0].cpu().numpy()
        # Map to labels
        labels = ['negative', 'neutral', 'positive'][: self.num_labels]
        return {
            'logits': logits[0].cpu().numpy(),
            'probs': {label: float(probs[i]) for i, label in enumerate(labels)},
            'label': labels[int(probs.argmax())],
        }


class TopicExtractor:
    """Extract salient topics or aspects from Arabic reviews.

    This class implements a simple aspect extraction pipeline tailored to
    product reviews.  It first attempts to run a part‑of‑speech (POS)
    tagger to identify nouns and adjectives that may constitute aspect
    phrases.  If a POS tagger is unavailable, it falls back to
    unsupervised keyword extraction via YAKE.  Identified phrases are
    cleaned and boosted according to a list of domain‑specific
    aspect terms.
    """

    #: Known product aspects to boost in scoring
    ASPECT_WORDS = {
        'منتج', 'الجودة', 'جودة', 'السعر', 'خدمة', 'خدمة العملاء', 'التوصيل',
        'الشحن', 'التغليف', 'العبوة', 'النكهة', 'الطعم', 'الحجم', 'اللون', 'المتانة'
    }

    def __init__(self, enable_pos_tagger: bool = True, top_k: int = 3) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.top_k = top_k
        self.pos_pipeline = None
        if enable_pos_tagger:
            try:
                from transformers import pipeline as hf_pipeline
                # Load a CAMeLBERT POS tagger (can be replaced with other models)
                model_id = 'CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa'
                self.pos_pipeline = hf_pipeline(
                    'token-classification', model=model_id, aggregation_strategy='simple'
                )
                self.logger.info("Loaded CAMeLBERT POS tagger for MSA")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load POS tagger; will use YAKE fallback: {e}"
                )
                self.pos_pipeline = None

    def _extract_pos_candidates(self, text: str) -> List[str]:
        """Use the POS tagger to extract noun/adjective phrases from the text."""
        if not self.pos_pipeline:
            return []
        try:
            results = self.pos_pipeline(text)
            phrases = []
            current = []
            for token_info in results:
                tag = token_info.get('entity', '').lower()
                token = token_info.get('word', '').strip()
                # CAMeL POS tags nouns with 'noun' and adjectives with 'adj'
                if tag.startswith('noun') or tag.startswith('adj'):
                    current.append(token)
                else:
                    if current:
                        phrases.append(' '.join(current))
                        current = []
            if current:
                phrases.append(' '.join(current))
            return phrases
        except Exception as e:
            self.logger.warning(f"POS tagging failed: {e}")
            return []

    def _extract_yake_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords via YAKE (fallback when POS tagger is unavailable)."""
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(
                lan='ar', n=2, top=self.top_k * 3, features=None
            )
            keywords = kw_extractor.extract_keywords(text)
            return keywords
        except Exception as e:
            self.logger.warning(f"YAKE extraction failed: {e}")
            return []

    @staticmethod
    def _clean_phrase(phrase: str) -> str:
        """Clean and normalise extracted phrase for consistency."""
        import re
        # Remove leading conjunctions
        phrase = re.sub(r"^(ولكن|لكن|و|بس)\s+", "", phrase)
        # Remove extraneous punctuation
        phrase = re.sub(r"[^\w\u0600-\u06FF\s]", "", phrase)
        # Collapse whitespace
        phrase = re.sub(r"\s+", " ", phrase).strip()
        return phrase

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract up to ``top_k`` salient topics/aspects from the input text.

        The method first normalises the text (basic normalisation), then
        attempts POS‑based extraction; if none are found, it falls back
        to YAKE.  Aspect words present in the extracted phrases are
        boosted slightly.
        """
        # Basic normalisation for topic extraction – we do not rely on
        # external libraries here to keep extraction lightweight
        text_norm = ArabicPreprocessor._basic_normalize(text)
        candidates: List[str] = self._extract_pos_candidates(text_norm)
        scores: Dict[str, float] = {}
        if not candidates:
            # Use YAKE fallback when POS tagging fails
            for phrase, raw_score in self._extract_yake_keywords(text_norm):
                cleaned = self._clean_phrase(phrase)
                if cleaned:
                    scores[cleaned] = max(scores.get(cleaned, 0.0), 1.0 - raw_score)
        else:
            # POS‑based candidates: assign a base score of 0.8
            for phrase in candidates:
                cleaned = self._clean_phrase(phrase)
                if cleaned:
                    scores[cleaned] = max(scores.get(cleaned, 0.0), 0.8)
        # Apply aspect boosting
        boosted: List[Tuple[str, float]] = []
        for phrase, score in scores.items():
            boost = 0.15 if any(aspect in phrase for aspect in self.ASPECT_WORDS) else 0.0
            boosted.append((phrase, min(1.0, score + boost)))
        # Sort by score descending and return top_k
        boosted.sort(key=lambda x: x[1], reverse=True)
        return [
            {"topic": phrase, "score": round(score, 4)}
            for phrase, score in boosted[: self.top_k]
        ]


@dataclass
class AnalysisResult:
    """Data class to hold the output of the sentiment analysis pipeline."""

    original_text: str
    cleaned_text: str
    normalized_text: str
    language: str
    dialect: Optional[str]
    sentiment: Dict[str, Any]
    topics: List[Dict[str, Any]]
    model_name: str


class SentimentPipeline:
    """End‑to‑end pipeline for analysing Arabic product reviews.

    The pipeline composes the pre‑processor, sentiment classifier and topic
    extractor into a single interface.  Clients call :meth:`analyze`
    with a raw review and receive structured output.  Dialect detection
    is optional and can be enabled during construction.
    """

    def __init__(
        self,
        sentiment_model: Optional[ArbMCNNBiClassifier] = None,
        model_name: str = "Arb‑MCNN‑Bi",
        enable_dialect_id: bool = True,
        enable_pos_tagger: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pre = ArabicPreprocessor(enable_dialect_id=enable_dialect_id)
        self.topics = TopicExtractor(enable_pos_tagger=enable_pos_tagger)
        if sentiment_model is None:
            # Lazily create a classifier; this may take a while if weights
            # need to be downloaded from Hugging Face.
            self.logger.info(
                "Initialising Arb‑MCNN‑Bi sentiment classifier (pre‑trained base only)"
            )
            self.classifier = ArbMCNNBiClassifier()
        else:
            self.classifier = sentiment_model
        self.model_name = model_name

    def detect_language(self, text: str) -> str:
        """Detect the primary language of the text using langdetect.

        Only Arabic (ar) and English (en) are supported. Defaults to 'ar' when 
        detection fails. Raises ValueError if an unsupported language is detected.
        This method is separated out to allow overriding or substituting with more
        specialised detectors.
        """
        try:
            from langdetect import detect
            detected_lang = detect(text)
            # Only accept Arabic and English
            if detected_lang in ["ar", "en"]:
                return detected_lang
            # If unsupported language detected, raise error
            self.logger.warning(f"Unsupported language detected: {detected_lang}. Only Arabic (ar) and English (en) are allowed.")
            raise ValueError(f"Unsupported language detected: {detected_lang}. Only Arabic and English are allowed.")
        except ValueError:
            # Re-raise ValueError for unsupported languages
            raise
        except Exception:
            return 'ar'

    def analyze(self, text: str) -> AnalysisResult:
        """Analyse a raw review and return a structured result.

        Parameters
        ----------
        text : str
            Raw review text.

        Returns
        -------
        AnalysisResult
            Structured output containing normalised text, detected
            language/dialect, sentiment scores and extracted topics.
        """
        # Keep original for reporting
        original = text
        # Clean whitespace and control characters
        cleaned = text.strip()
        # Detect language (fallback to Arabic)
        try:
            lang = self.detect_language(cleaned) or 'ar'
        except ValueError as e:
            # Re-raise ValueError for unsupported languages
            self.logger.error(f"Language validation failed: {e}")
            raise
        # Normalise
        normalized = self.pre.normalize(cleaned)
        # Detect dialect if enabled and the language is Arabic
        dialect = None
        if lang == 'ar':
            dialect = self.pre.detect_dialect(normalized)
        # Perform sentiment prediction
        sentiment_out = self.classifier.predict(normalized)
        # Extract topics
        topics = self.topics.extract(normalized)
        return AnalysisResult(
            original_text=original,
            cleaned_text=cleaned,
            normalized_text=normalized,
            language=lang,
            dialect=dialect,
            sentiment=sentiment_out,
            topics=topics,
            model_name=self.model_name,
        )


def _demo() -> None:
    """Simple demonstration of the sentiment pipeline.

    This function can be run directly (e.g. ``python arabic_sentiment_pipeline.py``)
    to see the pipeline in action.  It processes a sample product review
    and prints the analysis result.  Note that large models may need to
    download weights the first time this is run.
    """
    logging.basicConfig(level=logging.INFO)
    review = "استمتعت بالمنتج جداً، ولكن الشحن كان بطيئاً"
    pipeline = SentimentPipeline()
    result = pipeline.analyze(review)
    from pprint import pprint
    pprint(result)


if __name__ == '__main__':
    _demo()