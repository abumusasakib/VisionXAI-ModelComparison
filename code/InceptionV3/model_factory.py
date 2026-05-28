"""InceptionV3 model factory that reconstructs the encoder+decoder used in the
notebook and provides a `ModelWrapper` with `load_checkpoint()`, `generate()`,
and `to()` methods.

This factory attempts to:
- load a tokenizer from common locations (results/tokenizer.pkl, /data/tokenizer.pkl)
- build an InceptionV3 feature extractor
- build a small attention+decoder stack matching the notebook structure
- load weights from TensorFlow checkpoints when possible
- fall back to loading `predictions.json` if checkpoints aren't compatible
"""
from typing import Tuple, Optional
from pathlib import Path
import json
import os
import numpy as np

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def _find_tokenizer():
    import pickle
    candidates = [
        Path("./results/tokenizer.pkl"),
        Path("results/tokenizer.pkl"),
        Path("/results/tokenizer.pkl"),
        Path("/data/InceptionV3/tokenizer.pkl"),
        Path("data/InceptionV3/tokenizer.pkl"),
        Path("/data/tokenizer.pkl"),
        Path("data/tokenizer.pkl"),
        Path("tokenizer.pkl"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    tok = pickle.load(f)
                try:
                    logger.info(f"Found tokenizer at {p}")
                except Exception:
                    pass
                return tok
            except Exception:
                continue
    return None


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim=256):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, units=512):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def reset_state(self, batch_size=1):
        return tf.zeros((batch_size, self.units))

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights


class ModelWrapper:
    def __init__(self):
        # Build raw InceptionV3 extractor
        base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        self.inception_extractor = tf.keras.Model(base.input, base.layers[-1].output)
        # Build encoder (CNN_Encoder matching checkpoint variable names)
        self.encoder = CNN_Encoder(256)
        # default params
        self.tokenizer = _find_tokenizer()
        # create decoder lazily in load_checkpoint to match checkpoint vocab size
        self.decoder = None
        # optimizer included so tf.train.Checkpoint can restore optimizer state
        self.optimizer = tf.keras.optimizers.Adam()
        self.preds = {}

    def to(self, device: str):
        # no-op for TF models
        return

    def _preprocess_image(self, path: str):
        import cv2

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img.astype(np.float32))
        return np.expand_dims(img, 0)

    def load_checkpoint(self, path: str):
        p = Path(path)
        logger.info(f"load_checkpoint called with path={path} exists={p.exists()}")
        # Attempt to infer vocab size from tokenizer and checkpoint before creating decoder
        tok_size = (len(self.tokenizer.word_index) + 1) if self.tokenizer else 10001
        ckpt_vocab = None
        ckpt_prefix = None
        # determine candidate checkpoint prefixes
        candidates = []
        if p.is_dir():
            latest = tf.train.latest_checkpoint(str(p))
            if latest:
                candidates.append(latest)
        else:
            if p.suffix == '.index':
                candidates.append(str(p)[:-len('.index')])
            if 'ckpt-' in p.name:
                candidates.append(str(path))
            parent_latest = tf.train.latest_checkpoint(str(p.parent))
            if parent_latest:
                candidates.append(parent_latest)

        for cand in candidates:
            try:
                vars_list = tf.train.list_variables(cand)
                for name, shape in vars_list:
                    lname = name.lower()
                    # look for embedding-like variable to infer vocab size, ignoring position embeddings
                    if 'emb' in lname and 'position' not in lname and len(shape) >= 1:
                        ckpt_vocab = int(shape[0])
                        ckpt_prefix = cand
                        break
                if ckpt_vocab is not None:
                    logger.info(f"Inferred checkpoint vocab size {ckpt_vocab} from {ckpt_prefix}")
                    break
            except Exception:
                continue

        vocab_size = ckpt_vocab if ckpt_vocab is not None else tok_size
        if self.decoder is None:
            self.decoder = RNN_Decoder(vocab_size=vocab_size, embedding_dim=256, units=512)

        # Now create the checkpoint object and attempt restore
        try:
            ck = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
            # prefer explicit candidate prefixes discovered above
            if ckpt_prefix is not None:
                try:
                    status = ck.restore(ckpt_prefix)
                    try:
                        status.expect_partial()
                        logger.info(f"Restore from {ckpt_prefix} called expect_partial()")
                    except Exception as e:
                        logger.warning(f"expect_partial() raised for {ckpt_prefix}: {e}")
                except Exception as e:
                    logger.warning(f"Checkpoint restore failed for {ckpt_prefix}: {e}")
                # after restore, dump diagnostic snapshot
                self._dump_decoder_snapshot()
                return
            if p.is_dir():
                latest = tf.train.latest_checkpoint(str(p))
                if latest:
                    try:
                        status = ck.restore(latest)
                        try:
                            status.expect_partial()
                            logger.info(f"Restore from {latest} called expect_partial()")
                        except Exception as e:
                            logger.warning(f"expect_partial() raised for {latest}: {e}")
                    except Exception as e:
                        logger.warning(f"Checkpoint restore failed for {latest}: {e}")
                    self._dump_decoder_snapshot()
                    return
            else:
                if p.suffix == '.index':
                    candidate = str(p)[:-len('.index')]
                    try:
                        status = ck.restore(candidate)
                        try:
                            status.expect_partial()
                            logger.info(f"Restore from {candidate} called expect_partial()")
                        except Exception as e:
                            logger.warning(f"expect_partial() raised for {candidate}: {e}")
                    except Exception as e:
                        logger.warning(f"Checkpoint restore failed for {candidate}: {e}")
                    self._dump_decoder_snapshot()
                    return
                if 'ckpt-' in p.name:
                    try:
                        status = ck.restore(str(path))
                        try:
                            status.expect_partial()
                            logger.info(f"Restore from {path} called expect_partial()")
                        except Exception as e:
                            logger.warning(f"expect_partial() raised for {path}: {e}")
                    except Exception as e:
                        logger.warning(f"Checkpoint restore failed for {path}: {e}")
                    self._dump_decoder_snapshot()
                    return
                parent_latest = tf.train.latest_checkpoint(str(p.parent))
                if parent_latest:
                    try:
                        status = ck.restore(parent_latest)
                        try:
                            status.expect_partial()
                            logger.info(f"Restore from {parent_latest} called expect_partial()")
                        except Exception as e:
                            logger.warning(f"expect_partial() raised for {parent_latest}: {e}")
                    except Exception as e:
                        logger.warning(f"Checkpoint restore failed for {parent_latest}: {e}")
                    self._dump_decoder_snapshot()
                    return
        except Exception:
            pass

        # If path is directory, search for predictions.json
        if p.is_dir():
            for name in ("predictions.json", "preds.json", "captions.json", "generated_captions.json"):
                f = p / name
                if f.exists():
                    try:
                        self.preds = json.loads(f.read_text(encoding='utf-8'))
                        logger.info(f"Loaded precomputed preds from {f} (count={len(self.preds) if isinstance(self.preds, dict) else 'unknown'})")
                        return
                    except Exception:
                        continue

        # If path is file and JSON
        if p.is_file() and p.suffix in ('.json',):
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    self.preds = data
                    logger.info(f"Loaded JSON preds file {p} (count={len(self.preds)})")
                    return
            except Exception:
                pass

    def _dump_decoder_snapshot(self):
        try:
            if self.decoder is None:
                logger.info("No decoder to dump snapshot for")
                return
            emb_w = None
            try:
                emb_w = self.decoder.embedding.get_weights()[0]
            except Exception:
                emb_w = None
            if emb_w is not None:
                logger.info(f"Decoder embedding shape: {emb_w.shape}")
                # log first 5 token vectors sum and norms as a tiny fingerprint
                sample = emb_w[:5]
                norms = (np.linalg.norm(sample, axis=1)).tolist()
                sums = (sample.sum(axis=1)).tolist()
                logger.info(f"Embedding sample norms={norms} sums={sums}")
                # If tokenizer present, log embeddings for key tokens and the first few tokens
                try:
                    tok = self.tokenizer
                    if tok is not None:
                        # try important special tokens
                        keys = ['<start>', '<end>']
                        # add first 10 token ids from index_word
                        first_ids = []
                        try:
                            for i in range(1, min(11, max(int(k) for k in tok.index_word.keys()) + 1)):
                                if i in tok.index_word:
                                    first_ids.append(i)
                        except Exception:
                            # fallback: iterate index_word items
                            first_ids = list(sorted(tok.index_word.keys()))[:10]
                        # collect unique ids to inspect
                        ids = []
                        for k in keys:
                            vid = tok.word_index.get(k)
                            if vid is not None:
                                ids.append(vid)
                        ids.extend(x for x in first_ids if x not in ids)
                        ids = ids[:15]
                        for vid in ids:
                            word = tok.index_word.get(vid, '<UNK>')
                            if vid < emb_w.shape[0]:
                                vec = emb_w[vid]
                                logger.info(f"embed[idx={vid}] word={word} norm={np.linalg.norm(vec):.4f} sample={vec[:8].tolist()}")
                            else:
                                logger.info(f"embed idx {vid} >= vocab_size ({emb_w.shape[0]})")
                except Exception as e:
                    logger.warning(f"Failed tokenizer-embedding diagnostics: {e}")
            # fc2 (vocab projection) kernel
            try:
                fc2_w = self.decoder.fc2.get_weights()[0]
                logger.info(f"Decoder fc2 kernel shape: {fc2_w.shape}")
                logger.info(f"fc2 kernel sample row sums: {fc2_w[:5].sum(axis=1).tolist()}")
            except Exception:
                logger.info("Could not read decoder fc2 weights")
        except Exception as e:
            logger.warning(f"_dump_decoder_snapshot failed: {e}")

    def generate(self, img_id: Optional[str] = None, decode: str = "greedy", beam_size: int = 3, tokenizer=None):
        # If precomputed preds exist, return them
        key = str(img_id)
        if key in self.preds:
            return {"caption": self.preds[key]}

        # determine tokenizer
        tok = tokenizer or self.tokenizer
        if tok is None:
            logger.warning(f"No tokenizer available in generate() for img_id={img_id}")
            return {"caption": ""}

        # log tokenizer indices
        try:
            start_idx = tok.word_index.get('<start>')
            end_idx = tok.word_index.get('<end>')
        except Exception:
            start_idx = None
            end_idx = None
        logger.info(f"generate called img_id={img_id} tokenizer_start={start_idx} tokenizer_end={end_idx}")

        # resolve image path: annotations often provide filenames like '1.jpg'
        # while images live under mounted `/data/...`. Try to locate the file
        # under common data roots before preprocessing.
        img_path = Path(key)
        if not img_path.exists():
            # try common roots
            candidates = [Path('/data'), Path('data'), Path('.')]
            found = None
            for root in candidates:
                try:
                    # try exact filename first
                    for p in root.rglob(img_path.name):
                        if p.is_file():
                            found = p
                            break
                    if found:
                        break
                    # if not found, try matching by stem with any extension (e.g., '0' -> '0.jpg')
                    pattern = f"{img_path.name}.*"
                    for p in root.rglob(pattern):
                        if p.is_file():
                            found = p
                            break
                    if found:
                        break
                except Exception:
                    continue
                if found:
                    break
            if found:
                img_path = found
        logger.info(f"Resolved image path for {img_id} -> {img_path} exists={img_path.exists()}")

        img = self._preprocess_image(str(img_path))
        features = self.inception_extractor(img)  # shape (1, H, W, C)
        # flatten spatial dims to sequence
        features = tf.reshape(features, (features.shape[0], -1, features.shape[-1]))
        # project encoder features using restored CNN_Encoder
        features = self.encoder(features)

        # Greedy decoding
        result = []
        attention_plot = []
        hidden = self.decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([tok.word_index.get('<start>', 1)], 0)
        for i in range(50):
            predictions, hidden, attention = self.decoder(dec_input, features, hidden)
            attention_plot.append(tf.reshape(attention, (-1,)).numpy().tolist())
            
            # Find the best token index that is not in the blacklist
            preds_at_step = predictions[0]
            sampled_token_indices = np.argsort(preds_at_step.numpy())[::-1]
            predicted_id = None
            word = ''
            for token_idx in sampled_token_indices:
                w = tok.index_word.get(token_idx, '')
                if w not in ("[UNK]", "<pad>", "<start>"):
                    predicted_id = int(token_idx)
                    word = w
                    break
            
            if predicted_id is None or word == '<end>' or word == '':
                break
            result.append(word)
            dec_input = tf.expand_dims([predicted_id], 0)

        caption = ' '.join(result)
        return {"caption": caption, "attention": attention_plot}


def build_model() -> Tuple[ModelWrapper, Optional[object], str]:
    m = ModelWrapper()
    return m, m.tokenizer, 'cpu'
