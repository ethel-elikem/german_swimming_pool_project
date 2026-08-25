"""
nlp_baseline_classifier.py
═══════════════════════════════════════════════════════════════════════════════
NLP baseline pipeline for pool review classification.
English-language reviews of German swimming pools.

Classifies reviews on two binary dimensions:
  • social_encounter      (yes/no)  — does the review mention other people?
  • racial_connotations   (yes/no)  — does it use racial/ethnic/religious framing?

Three methods are implemented as a stacked comparison:

  METHOD 1 — Keyword / Rule-Based (zero-shot, no training data needed)
             Explainable lexicon built for English-language reviews of
             German pools. Hard lower bound. Every false negative is a
             theoretically interesting case of implicit/coded language.

  METHOD 2 — TF-IDF + Logistic Regression / Linear SVM
             Classical bag-of-words ML. Uses your LLM labels as silver
             labels (or a manually verified subset as gold labels).
             Character n-grams included to handle morphological variation.

  METHOD 3 — Transformer Fine-Tuning (roberta-base / bert-base-uncased)
             Contextual model. Captures implicit framing, nostalgia,
             and coded language far better than bag-of-words methods.

Theoretical anchoring of the lexicons
──────────────────────────────────────
  social_agents       → Goffman (1963): co-presence; any textual reference
                        to another person being present at the pool.

  racial_explicit     → van Dijk (1993): nomination strategy; direct
                        naming of racialised groups in English.

  racial_implicit     → Wodak (2015) recontextualisation + Bonilla-Silva
                        (2006) colour-blind racism: coded/euphemistic
                        language that carries racialised meaning in the
                        context of English-language discourse about
                        migration in Europe.

  religious_cultural  → Captures the burkini/swimwear controversy and
                        religious dress as ethnic/cultural boundary marker
                        (central to German pool politics 2015-2016).

  language_boundary   → Lamont (2002): cultural boundary via language;
                        van Dijk: linguistic othering.

  nostalgia_lament    → Lamont boundary lamenting; Wimmer (2008) boundary
                        contraction; Anderson (2011) comfort-level anxiety.

  spatial_ownership   → Puwar (2004) space invaders; Anderson cosmopolitan
                        canopy under stress.

NOTE on reviewer population
───────────────────────────
English-language reviewers of German pools are self-selected: predominantly
tourists, expats, and international visitors. This likely over-represents
Outsider/Dominant and Detached Observer positions, and under-represents
Insider/Affected voices (who more commonly write in German or Turkish).
Acknowledge this in your methods section.

Usage
─────
  # Method 1 only — no training data needed, runs instantly
  python nlp_baseline_classifier.py run \
      --input   reviews.parquet \
      --labels  llm_labels.csv \
      --methods rule \
      --output  baseline_results/

  # Full comparison — all three methods
  python nlp_baseline_classifier.py run \
      --input   reviews.parquet \
      --labels  llm_labels.csv \
      --methods rule tfidf bert \
      --output  baseline_results/

  # Gold-label sanity check on a manually verified subset
  python nlp_baseline_classifier.py run \
      --input   gold_subset.parquet \
      --labels  gold_labels.csv \
      --methods rule tfidf bert \
      --gold \
      --output  baseline_results/

Input format
────────────
  reviews.parquet  — columns: review_id, text  (+ optionally: summary)
  llm_labels.csv   — columns: review_id, social_encounter,
                      racial_connotations   (values: "yes" / "no")

Dependencies
────────────
  pip install pandas scikit-learn transformers torch tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score,
)
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = output_dir / f"baseline_{ts}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    fh = logging.FileHandler(log);         fh.setFormatter(fmt);  root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); root.addHandler(ch)
    logging.info(f"Log → {log}")


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH LEXICONS
# ══════════════════════════════════════════════════════════════════════════════

LEXICONS = {

    # ── Dimension 1: Social Encounter ─────────────────────────────────────
    # Operationalises Goffman (1963) co-presence: any textual reference
    # to another person being present at the pool — staff or other visitors.
    "social_agents": [
        # Staff references
        r"\bstaff\b",           r"\blifeguard\w*",       r"\battendant\w*",
        r"\breceptionist\w*",   r"\bemployee\w*",        r"\bworker\w*",
        r"\bsecurity\b",        r"\bguard\b",            r"\bmanager\b",
        # Other visitors — generic
        r"\bpeople\b",          r"\bperson\b",           r"\bfolks\b",
        r"\bcrowd\w*",          r"\bvisitor\w*",         r"\bguest\w*",
        r"\bfamilies\b",        r"\bfamily\b",           r"\bchildren\b",
        r"\bkids\b",            r"\bteenager\w*",        r"\byouth\b",
        r"\bgroup\w*",          r"\bman\b",              r"\bmen\b",
        r"\bwoman\b",           r"\bwomen\b",            r"\bboy\w*",
        r"\bgirl\w*",           r"\bcouple\w*",
        # Crowd / atmosphere shaped by others
        r"\bpacked\b",          r"\bcrowded\b",          r"\bbustling\b",
        r"\bovercrowd\w*",      r"\bbehaviou?r\b",       r"\bconduct\b",
        r"\batmosphere\b",      r"\bvibe\b",             r"\bnoisy\b",
        r"\bnoise\b",           r"\bchaotic\b",          r"\brough\b",
        # Interaction / observation verbs
        r"\bstaring\b",         r"\bwatching\b",         r"\bapproached\b",
        r"\bharrassed\b",       r"\bfollowed\b",         r"\byelled\b",
        r"\btold (to|me|us)\b",
    ],

    # ── Dimension 2a: Racial/Ethnic — Explicit naming ─────────────────────
    # Operationalises van Dijk (1993) "nomination" strategy.
    # Direct naming of racialised groups in English.
    "racial_explicit": [
        r"\bforeigner\w*",       r"\bimmigrant\w*",       r"\bmigrant\w*",
        r"\brefugee\w*",         r"\basylu\w*",
        r"\barabs?\b",           r"\barabic\b",
        r"\bturk\w*",            r"\bturkish\b",
        r"\bafghan\w*",          r"\bsyrian\w*",
        r"\bpakistani\w*",       r"\bnorth african\w*",   r"\bafrican\w*",
        r"\bromanian\w*",        r"\bbulgarian\w*",
        r"\beast european\w*",   r"\beastern european\w*",
        r"\bnon.?german\w*",     r"\bnon.?western\w*",
        r"\bforeign\w*",
        r"\bblack (people|men|women|guys)\b",
        r"\bwhite (people|men|women|guys)\b",
        r"\bpoc\b",              r"\bpeople of colo(u)?r\b",
    ],

    # ── Dimension 2b: Racial/Ethnic — Implicit / Coded / Proxy ───────────
    # Operationalises Wodak (2015) recontextualisation and Bonilla-Silva
    # (2006) colour-blind racism. These patterns carry racialised meaning
    # without explicitly naming groups. Every FN here is theoretically rich.
    "racial_implicit": [
        # Nostalgia / displacement — boundary lamenting (Lamont / Wimmer)
        r"\bnot (what|how) it used to be\b",
        r"\bused to be (nicer|better|cleaner|great|lovely|fine|safe)\b",
        r"\bhas (really )?changed\b",
        r"\bnot the same (any ?more|crowd|clientele|atmosphere)\b",
        r"\bused to (come|go|enjoy|love)\b.{0,40}\b(but|now|anymore)\b",
        r"\bcompletely (different|changed|transformed)\b",
        r"\bno longer\b",
        # "Certain clientele" euphemism formulas
        r"\bcertain (type|kind|sort) of (people|crowd|clientele|visitors)\b",
        r"\ba (particular|specific) (crowd|type|clientele)\b",
        r"\bthe (crowd|clientele|people) (has|have) changed\b",
        r"\bnot (really )?my (kind of )?(place|crowd|pool)\b",
        # Numerical / statistical displacement framing (van Dijk)
        r"\boverwhelm\w*",       r"\boverrun\b",
        r"\bhardly any (german\w*|local\w*|european\w*)\b",
        r"\bmostly\b.{0,20}\b(foreign\w*|migrant\w*|arab\w*|turkish)\b",
        r"\bpredominant\w*\b.{0,20}\b(foreign\w*|migrant\w*)\b",
        r"\bvery (few|little).{0,15}\b(german\w*|local\w*|european\w*)\b",
    ],

    # ── Dimension 2c: Religious / Cultural Dress ──────────────────────────
    # Captures the burkini controversy central to German pool politics.
    # In English reviews often framed as a rules/enforcement complaint.
    "religious_cultural": [
        r"\bbur?kini\b",         r"\bhijab\b",            r"\bniqab\b",
        r"\bheadscarf\b",        r"\bveil\w*",            r"\bmodest\w* swimwear\b",
        r"\bfull(y)? covered\b", r"\bfully clothed\b",
        r"\bclothes in (the )?pool\b",
        r"\bclothes in (the )?water\b",
        r"\bswimming (in|with) clothes\b",
        r"\bfull (length|coverage) (swimsuit|swimwear|clothing)\b",
        r"\bwearing (clothes|clothing|a dress|trousers) (in|into) (the )?pool\b",
        r"\bmuslim\w*",          r"\bislamic?\b",
        r"\breligious (dress|clothing|swimwear|garment)\b",
        r"\bprayer\b",           r"\bpray\b",             r"\bmosque\b",
    ],

    # ── Dimension 2d: Language as Ethnic Boundary ─────────────────────────
    # Lamont (2002) cultural boundary via language; van Dijk linguistic
    # othering. In English reviews of German pools, complaints about
    # non-German-speaking visitors are racialised boundary markers;
    # complaints about staff not speaking English are more ambiguous
    # (tourist frustration vs. cultural othering — both are flagged here
    # for qualitative review).
    "language_boundary": [
        r"\bno (one |one )?speaks? english\b",
        r"\bhardly (anyone|any staff) speaks? english\b",
        r"\blanguage barrier\b",
        r"\bcould(n'?t| not) communicate\b",
        r"\bcould(n'?t| not) understand (anyone|them|the staff)\b",
        r"\bno(body| one) spoke? (any )?german\b",
        r"\bno(body| one) (here )?speaks? german\b",
        r"\bhardly (any|anyone) (spoke|speaks|speaking) german\b",
        r"\bonly (arabic|turkish|russian|romanian)\b",
        r"\bno (english|german) spoken\b",
        r"\bcouldn'?t hear (any )?(english|german)\b",
        r"\bdidn'?t (understand|speak) the language\b",
    ],

    # ── Dimension 2e: Nostalgia / Spatial Lament ──────────────────────────
    # Lamont boundary lamenting; Wimmer (2008) boundary contraction.
    # Reviews expressing that the space has deteriorated, become unsafe,
    # or is no longer family-friendly enact racialised spatial ownership
    # (Puwar 2004) even without naming a group. High false-positive risk —
    # every match needs qualitative review.
    "nostalgia_lament": [
        r"\bused to (be|love|enjoy|come)\b",
        r"\bwent downhill\b",     r"\bgone downhill\b",
        r"\bnot what (it|they) used to be\b",
        r"\bnot as (good|nice|clean|safe|family.friendly) as (it|before)\b",
        r"\bhas (really )?changed (for the worse)?\b",
        r"\bused to be (a great|an amazing|a wonderful|a lovely|a nice|a good)\b",
        r"\bpast (few|couple of) years\b.{0,40}\bworse\b",
        r"\byou (can'?t|cannot|could not) (enjoy|relax|swim) (any ?more|anymore)\b",
        r"\bdon'?t (feel safe|feel welcome|feel comfortable) (any ?more|anymore)?\b",
        r"\bfeel (unsafe|unwelcome|uncomfortable) (now|these days|anymore)\b",
        r"\bnot (family.friendly|safe for families) (any ?more|anymore)?\b",
    ],

    # ── Dimension 2f: Spatial Ownership / Belonging ───────────────────────
    # Puwar (2004) "space invaders"; Anderson (2011) cosmopolitan canopy
    # under stress: who is constructed as belonging, whose presence is
    # marked as Other.
    "spatial_ownership": [
        r"\b(my|our) (local |favourite |neighborhood )pool\b",
        r"\bbelongs? to (everyone|the community|locals)\b",
        r"\bfeel like (a|an) (outsider|stranger|minority)\b",
        r"\bfeel (out of place|unwelcome|uncomfortable)\b",
        r"\bdidn'?t (belong|fit in)\b",
        r"\bright to (be here|use this|swim here)\b",
        r"\bstared (at me|at us)\b",
        r"\bwatched (constantly|the whole time|us)\b",
        r"\bstaring (at me|at us|at my|at our)\b",
        r"\basked (to leave|us to leave|me to leave)\b",
        r"\btold (to leave|to change|me to leave|us to leave)\b",
        r"\bkicked (out|us out|me out)\b",
        r"\bsegregat\w*",
        r"\bdiscriminat\w*",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: RULE-BASED CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedClassifier:
    """
    Keyword / regex lexicon classifier for English pool reviews.

    Decision cascade
    ────────────────
    social_encounter = yes   iff   any social_agents pattern fires.
    racial_connotations = yes  iff  social_encounter=yes  AND  any pattern
      fires in: racial_explicit | racial_implicit | religious_cultural |
                language_boundary | nostalgia_lament | spatial_ownership

    The cascade mirrors the codebook's gate-keeping logic.

    Output
    ──────
    Returns which rules fired for each review — enables direct qualitative
    inspection and iterative lexicon refinement. Every FN on racial_
    connotations is a review where racialised meaning was conveyed without
    any lexical trigger — these are your theoretically richest cases.
    """

    RACIAL_CATS = [
        "racial_explicit", "racial_implicit", "religious_cultural",
        "language_boundary", "nostalgia_lament", "spatial_ownership",
    ]

    def __init__(self, lexicons: dict):
        self.patterns = {
            cat: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in pats]
            for cat, pats in lexicons.items()
        }

    def _match(self, text: str, category: str) -> list[str]:
        return [m.group(0) for pat in self.patterns[category]
                if (m := pat.search(text))]

    def predict_one(self, text: str) -> dict:
        t = text.lower()
        se_matches = self._match(t, "social_agents")
        social_enc = "yes" if se_matches else "no"
        racial_matches: dict = {}
        if social_enc == "yes":
            for cat in self.RACIAL_CATS:
                m = self._match(t, cat)
                if m:
                    racial_matches[cat] = m
        return {
            "social_encounter"   : social_enc,
            "racial_connotations": "yes" if racial_matches else "no",
            "rule_se_matches"    : se_matches[:5],
            "rule_rc_matches"    : {k: v[:3] for k, v in racial_matches.items()},
        }

    def predict(self, texts: list[str]) -> list[dict]:
        return [self.predict_one(t) for t in tqdm(texts, desc="Rule-based")]


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: TF-IDF + CLASSICAL ML
# ══════════════════════════════════════════════════════════════════════════════

class TfidfClassifier:
    """
    TF-IDF vectoriser + Logistic Regression and Linear SVM.

    Four variants:
      LR_word    word unigrams + bigrams + LR
      LR_char    character 2-5-grams + LR
      SVM_word   word n-grams + LinearSVC
      SVM_char   char n-grams + LinearSVC

    Char n-grams capture spelling variants and partial-word morphology;
    useful for informal English where reviewers may write "burkinis",
    "non-Germans", "overrun-by", etc.

    Evaluation: stratified 5-fold CV on silver labels.
    LR top-feature export allows qualitative sanity-check.
    """

    def __init__(self):
        char_ngram = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, max_features=80_000, sublinear_tf=True,
        )
        word_ngram = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=2, max_features=50_000, sublinear_tf=True,
        )
        self.models = {
            "LR_word" : Pipeline([("tfidf", word_ngram),
                                   ("clf",  LogisticRegression(
                                       C=1.0, max_iter=1000,
                                       class_weight="balanced", random_state=42))]),
            "LR_char" : Pipeline([("tfidf", char_ngram),
                                   ("clf",  LogisticRegression(
                                       C=1.0, max_iter=1000,
                                       class_weight="balanced", random_state=42))]),
            "SVM_word": Pipeline([("tfidf", word_ngram),
                                   ("clf",  LinearSVC(
                                       C=0.5, max_iter=2000,
                                       class_weight="balanced", random_state=42))]),
            "SVM_char": Pipeline([("tfidf", char_ngram),
                                   ("clf",  LinearSVC(
                                       C=0.5, max_iter=2000,
                                       class_weight="balanced", random_state=42))]),
        }

    def cv_predict(self, texts: list[str], labels: list[str],
                   task: str, cv: int = 5) -> dict[str, list]:
        logging.info(f"  TF-IDF {cv}-fold CV | task={task} | n={len(texts)}")
        y   = np.array(labels)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        out = {}
        for name, pipe in self.models.items():
            logging.info(f"    {name}…")
            pred    = cross_val_predict(pipe, texts, y, cv=skf, n_jobs=-1)
            out[name] = pred.tolist()
        return out

    def get_top_features(self, texts: list[str], labels: list[str],
                         model_name: str = "LR_word", n: int = 25) -> dict:
        if "LR" not in model_name:
            return {}
        pipe = self.models[model_name]
        pipe.fit(texts, np.array(labels))
        names = pipe.named_steps["tfidf"].get_feature_names_out()
        coef  = pipe.named_steps["clf"].coef_[0]
        return {
            "top_yes": [(names[i], round(float(coef[i]), 4))
                        for i in coef.argsort()[-n:][::-1]],
            "top_no" : [(names[i], round(float(coef[i]), 4))
                        for i in coef.argsort()[:n]],
        }


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: TRANSFORMER FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

class TransformerClassifier:
    """
    Fine-tuned English transformer for binary sequence classification.

    Model options
    ─────────────
    roberta-base (default)
        Best general-purpose English classification model. RoBERTa removes
        BERT's next-sentence-prediction objective and trains on more data,
        consistently outperforming BERT-base on classification tasks that
        require sensitivity to subtle lexical choices — exactly what the
        racial_connotations dimension demands (Liu et al., 2019).

    bert-base-uncased
        Lighter; use if GPU memory is constrained.

    cardiffnlp/twitter-roberta-base-sentiment-latest
        Trained on informal social text; worth trying if your reviews
        are colloquial or contain slang.

    Training: stratified 3-fold CV, 3 epochs, lr=2e-5, warmup_ratio=0.1.
    Standard hyperparameters for fine-tuning on small datasets
    (Devlin et al., 2019; Sun et al., 2019).
    """

    MODELS = {
        "roberta"        : "roberta-base",
        "bert"           : "bert-base-uncased",
        "twitter_roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    }

    def __init__(self, model_key: str = "roberta", max_length: int = 128,
                 batch_size: int = 16, epochs: int = 3, lr: float = 2e-5):
        self.model_name = self.MODELS.get(model_key, model_key)
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs     = epochs
        self.lr         = lr
        self._check_deps()

    @staticmethod
    def _check_deps():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "Transformer support requires:\n"
                "  pip install transformers torch\n"
                "To skip: use --methods rule tfidf"
            )

    def cv_predict(self, texts: list[str], labels: list[str],
                   task: str, cv: int = 3) -> list[str]:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            Trainer, TrainingArguments,
        )
        from torch.utils.data import Dataset

        logging.info(
            f"  Transformer CV ({cv}-fold) | task={task} | model={self.model_name}\n"
            f"  epochs={self.epochs} | batch={self.batch_size} | "
            f"lr={self.lr} | max_len={self.max_length}"
        )

        label2id = {"no": 0, "yes": 1}
        id2label = {0: "no", 1: "yes"}
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        y   = np.array([label2id[l] for l in labels])
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        oof = np.zeros(len(texts), dtype=int)

        class ReviewDataset(Dataset):
            def __init__(self, enc, labs=None):
                self.enc  = enc
                self.labs = labs
            def __len__(self):
                return len(self.enc["input_ids"])
            def __getitem__(self, i):
                item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
                if self.labs is not None:
                    item["labels"] = torch.tensor(self.labs[i])
                return item

        for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, y)):
            logging.info(f"    Fold {fold+1}/{cv}")
            tr_enc  = tokenizer([texts[i] for i in tr_idx],  truncation=True,
                                padding=True, max_length=self.max_length)
            val_enc = tokenizer([texts[i] for i in val_idx], truncation=True,
                                padding=True, max_length=self.max_length)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2,
                id2label=id2label, label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            args = TrainingArguments(
                output_dir              = f"/tmp/transformer_fold_{fold}",
                num_train_epochs        = self.epochs,
                per_device_train_batch_size = self.batch_size,
                per_device_eval_batch_size  = self.batch_size * 2,
                learning_rate           = self.lr,
                warmup_ratio            = 0.1,
                weight_decay            = 0.01,
                logging_steps           = 50,
                save_strategy           = "no",
                seed                    = 42,
                fp16                    = torch.cuda.is_available(),
                report_to               = "none",
            )
            trainer = Trainer(
                model=model, args=args,
                train_dataset=ReviewDataset(tr_enc, y[tr_idx].tolist()),
            )
            trainer.train()
            raw          = trainer.predict(ReviewDataset(val_enc))
            oof[val_idx] = np.argmax(raw.predictions, axis=1)
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return [id2label[p] for p in oof]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: list, y_pred: list, method: str, task: str) -> dict:
    labels = ["no", "yes"]
    f1_mac = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    f1_yes = f1_score(y_true, y_pred, pos_label="yes",    average="binary", zero_division=0)
    prec_y = precision_score(y_true, y_pred, pos_label="yes", average="binary", zero_division=0)
    rec_y  = recall_score(y_true, y_pred, pos_label="yes",    average="binary", zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    sep = "─" * 64
    logging.info(f"\n{sep}")
    logging.info(f"  {method:<30} | task = {task}")
    logging.info(sep)
    logging.info(f"  Accuracy       : {acc:.4f}")
    logging.info(f"  F1 macro       : {f1_mac:.4f}")
    logging.info(f"  F1 (yes)       : {f1_yes:.4f}   Prec={prec_y:.4f}   Rec={rec_y:.4f}")
    logging.info(f"  Confusion matrix (rows=true, cols=pred)  [no, yes]")
    logging.info(f"                 pred-no   pred-yes")
    for i, row in enumerate(cm):
        logging.info(f"  true-{labels[i]}  :    {row[0]:5d}     {row[1]:5d}")
    logging.info(sep)

    return {
        "method": method, "task": task,
        "accuracy": round(acc, 4), "f1_macro": round(f1_mac, 4),
        "f1_yes": round(f1_yes, 4), "precision_yes": round(prec_y, 4),
        "recall_yes": round(rec_y, 4), "confusion_matrix": cm,
    }


def log_comparison_table(results: list[dict]) -> None:
    sep = "═" * 84
    logging.info(f"\n{sep}")
    logging.info("  BASELINE COMPARISON TABLE")
    logging.info(f"  {'Method':<30} {'Task':<24} {'Acc':>6} "
                 f"{'F1-mac':>7} {'F1-yes':>7} {'Prec':>7} {'Rec':>7}")
    logging.info("  " + "─"*82)
    for r in sorted(results, key=lambda x: (x["task"], -x["f1_macro"])):
        logging.info(
            f"  {r['method']:<30} {r['task']:<24} "
            f"{r['accuracy']:>6.4f} {r['f1_macro']:>7.4f} "
            f"{r['f1_yes']:>7.4f} {r['precision_yes']:>7.4f} {r['recall_yes']:>7.4f}"
        )
    logging.info(sep)
    logging.info(
        "\n  INTERPRETATION GUIDE\n"
        "  ─────────────────────────────────────────────────────────────────\n"
        "  • social_encounter is easier: expect rule-based ~0.80+, RoBERTa ~0.90+\n"
        "  • racial_connotations is harder: a large gap between rule-based\n"
        "    and RoBERTa is your empirical evidence that implicit/coded language\n"
        "    is doing the work — it cannot be captured by keyword matching alone.\n"
        "    This gap is publishable in its own right.\n"
        "  • Inspect errors_racial_connotations_Rule-Based.csv: every FN is\n"
        "    a review where racialised meaning was conveyed without any\n"
        "    lexical trigger. Qualitative analysis of these cases is the\n"
        "    most theoretically rich output of the baseline comparison.\n"
        "  ─────────────────────────────────────────────────────────────────"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def error_analysis(texts, ids, y_true, y_pred,
                   method, task, output_dir, n=100):
    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != yp:
            rows.append({
                "review_id" : ids[i],
                "text"      : texts[i][:400],
                "true_label": yt,
                "pred_label": yp,
                "error_type": "FP" if yp == "yes" else "FN",
            })
    if not rows:
        logging.info(f"  No errors: {method} / {task}")
        return
    slug = method.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    path = output_dir / f"errors_{task}_{slug}.csv"
    pd.DataFrame(rows).head(n).to_csv(path, index=False)
    fp = sum(1 for r in rows if r["error_type"] == "FP")
    fn = sum(1 for r in rows if r["error_type"] == "FN")
    logging.info(f"  Errors: {len(rows)} | FP={fp} FN={fn} → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(input_path: Path, labels_path: Path, text_col: str = "text"):
    logging.info(f"Loading reviews  : {input_path}")
    df  = (pd.read_parquet(input_path) if input_path.suffix == ".parquet"
           else pd.read_csv(input_path))
    df["review_id"] = df["review_id"].astype(str)

    logging.info(f"Loading labels   : {labels_path}")
    ldf = pd.read_csv(labels_path)
    ldf["review_id"] = ldf["review_id"].astype(str)

    merged = df.merge(
        ldf[["review_id", "social_encounter", "racial_connotations"]],
        on="review_id", how="inner"
    )
    if "summary" in merged.columns:
        merged["_text"] = merged["summary"].where(
            merged["summary"].notna() & (merged["summary"].str.strip() != ""),
            merged[text_col]
        )
    else:
        merged["_text"] = merged[text_col]

    merged = merged.dropna(subset=["_text", "social_encounter", "racial_connotations"])
    merged = merged[merged["social_encounter"].isin(["yes", "no"])]
    merged = merged[merged["racial_connotations"].isin(["yes", "no"])]

    logging.info(f"  {len(merged):,} reviews with valid labels")
    for col in ["social_encounter", "racial_connotations"]:
        vc      = merged[col].value_counts().to_dict()
        pct_yes = vc.get("yes", 0) / len(merged) * 100
        logging.info(f"  {col}: {vc}  ({pct_yes:.1f}% positive)")

    return (
        merged["_text"].tolist(),
        merged["review_id"].tolist(),
        merged["social_encounter"].tolist(),
        merged["racial_connotations"].tolist(),
        merged,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(args) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    texts, ids, se_labels, rc_labels, _ = load_data(
        Path(args.input), Path(args.labels), text_col=args.text_col
    )
    methods     = set(args.methods)
    all_results : list[dict] = []
    tasks       = {
        "social_encounter"   : se_labels,
        "racial_connotations": rc_labels,
    }

    # ── METHOD 1: RULE-BASED ──────────────────────────────────────────────
    if "rule" in methods:
        logging.info("\n" + "═"*64)
        logging.info("METHOD 1: Rule-Based Classifier (English lexicon)")
        logging.info("═"*64)
        rule_clf  = RuleBasedClassifier(LEXICONS)
        raw_preds = rule_clf.predict(texts)

        for task, true_labels in tasks.items():
            pred_labels = [p[task] for p in raw_preds]
            all_results.append(evaluate(true_labels, pred_labels, "Rule-Based", task))
            error_analysis(texts, ids, true_labels, pred_labels,
                           "Rule-Based", task, output_dir)

        # Full evidence file — invaluable for lexicon refinement
        evidence = [{
            "review_id"   : ids[i],
            "text_snippet": texts[i][:300],
            "se_pred"     : p["social_encounter"],
            "rc_pred"     : p["racial_connotations"],
            "se_true"     : se_labels[i],
            "rc_true"     : rc_labels[i],
            "se_correct"  : p["social_encounter"] == se_labels[i],
            "rc_correct"  : p["racial_connotations"] == rc_labels[i],
            "se_matches"  : " | ".join(p["rule_se_matches"]),
            "rc_matches"  : json.dumps(p["rule_rc_matches"], ensure_ascii=False),
        } for i, p in enumerate(raw_preds)]
        ev_path = output_dir / "rule_evidence.csv"
        pd.DataFrame(evidence).to_csv(ev_path, index=False)
        logging.info(f"  Rule match evidence → {ev_path.name}")

    # ── METHOD 2: TF-IDF ──────────────────────────────────────────────────
    if "tfidf" in methods:
        logging.info("\n" + "═"*64)
        logging.info("METHOD 2: TF-IDF + Classical ML")
        logging.info("═"*64)
        tfidf_clf = TfidfClassifier()
        for task, true_labels in tasks.items():
            logging.info(f"\n  Task: {task}")
            cv_preds = tfidf_clf.cv_predict(texts, true_labels, task)
            for model_name, pred_labels in cv_preds.items():
                all_results.append(
                    evaluate(true_labels, pred_labels, f"TF-IDF {model_name}", task)
                )
                error_analysis(texts, ids, true_labels, pred_labels,
                               f"TF-IDF {model_name}", task, output_dir)
            feats = tfidf_clf.get_top_features(texts, true_labels, n=25)
            if feats:
                logging.info(f"  Top 'yes' features: {[f[0] for f in feats['top_yes'][:12]]}")
                logging.info(f"  Top 'no'  features: {[f[0] for f in feats['top_no'][:12]]}")
                with open(output_dir / f"tfidf_features_{task}.json", "w") as f:
                    json.dump(feats, f, ensure_ascii=False, indent=2)

    # ── METHOD 3: TRANSFORMER ─────────────────────────────────────────────
    if "bert" in methods:
        logging.info("\n" + "═"*64)
        logging.info(f"METHOD 3: Transformer ({args.bert_model})")
        logging.info("═"*64)
        trans_clf = TransformerClassifier(
            model_key  = args.bert_model,
            max_length = args.bert_maxlen,
            batch_size = args.bert_batch,
            epochs     = args.bert_epochs,
        )
        for task, true_labels in tasks.items():
            logging.info(f"\n  Task: {task}")
            pred_labels = trans_clf.cv_predict(
                texts, true_labels, task, cv=args.bert_cv
            )
            label = f"Transformer ({args.bert_model})"
            all_results.append(evaluate(true_labels, pred_labels, label, task))
            error_analysis(texts, ids, true_labels, pred_labels,
                           label, task, output_dir)

    # ── SUMMARY ───────────────────────────────────────────────────────────
    if all_results:
        log_comparison_table(all_results)
        out = {
            "results"     : all_results,
            "generated_at": datetime.now().isoformat(),
            "n_reviews"   : len(texts),
            "label_dist"  : {
                "social_encounter_yes"   : se_labels.count("yes"),
                "social_encounter_no"    : se_labels.count("no"),
                "racial_connotations_yes": rc_labels.count("yes"),
                "racial_connotations_no" : rc_labels.count("no"),
            },
            "notes": (
                "english_reviews_german_pools. "
                "silver_labels_from_llm unless --gold used. "
                "FN analysis on racial_connotations is theoretically most valuable."
            ),
        }
        path = output_dir / "baseline_comparison.json"
        with open(path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logging.info(f"\nFull results → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        prog="nlp_baseline_classifier",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the baseline pipeline")
    run.add_argument("--input",    required=True,
                     help="Path to reviews (.parquet or .csv)")
    run.add_argument("--labels",   required=True,
                     help="Path to LLM labels CSV (review_id, "
                          "social_encounter, racial_connotations)")
    run.add_argument("--output",   default="baseline_results/")
    run.add_argument("--text-col", default="text", dest="text_col")
    run.add_argument("--gold",     action="store_true",
                     help="Labels are manually verified gold")
    run.add_argument(
        "--methods", nargs="+", default=["rule", "tfidf", "bert"],
        choices=["rule", "tfidf", "bert"],
        help="Methods to run. Use --methods rule for instant results.",
    )
    run.add_argument(
        "--bert-model", default="roberta",
        choices=["roberta", "bert", "twitter_roberta"],
        help=(
            "roberta          → roberta-base (recommended)\n"
            "bert             → bert-base-uncased (lighter)\n"
            "twitter_roberta  → cardiffnlp model (informal text)"
        ),
    )
    run.add_argument("--bert-epochs", type=int, default=3)
    run.add_argument("--bert-batch",  type=int, default=16)
    run.add_argument("--bert-maxlen", type=int, default=128)
    run.add_argument("--bert-cv",     type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "run":
        run_baseline(args)


if __name__ == "__main__":
    main()