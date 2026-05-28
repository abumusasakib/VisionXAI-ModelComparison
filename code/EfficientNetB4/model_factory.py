"""EfficientNetB4 model factory: build encoder+decoder compatible with notebooks."""
from typing import Tuple, Optional
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _find_tokenizer():
    import pickle
    candidates = [
        Path("./results/tokenizer.pkl"),
        Path("results/tokenizer.pkl"),
        Path("/data/tokenizer.pkl"),
        Path("tokenizer.pkl")
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    tok = pickle.load(f)
                return tok
            except Exception:
                continue
    return None

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")
        self.dense_2 = layers.Dense(embed_dim)
        self.dropout_1 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = self.layernorm_1(inputs)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        out_2 = self.dense_1(out_1)
        out_2 = self.dropout_1(out_2, training=training)
        out_2 = self.dense_2(out_2)
        return out_1 + out_2

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim=2048, num_heads=4, dropout_rate=0.3, vocab_size=2848, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(
            embed_dim=embed_dim, sequence_length=8, vocab_size=vocab_size
        )
        self.out = layers.Dense(vocab_size, activation="softmax")
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training=False, mask=None, return_attention=False):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2, attention_scores = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            training=training,
            return_attention_scores=True
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        if return_attention:
            return preds, attention_scores
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

def get_cnn_model():
    base_model = tf.keras.applications.EfficientNetB4(
        input_shape=(380, 380, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.GlobalAveragePooling2D()(base_model_out)
    base_model_out = layers.Dense(768, activation="relu")(base_model_out)
    base_model_out = layers.Dense(768, dtype="float32")(base_model_out)
    return keras.models.Model(base_model.input, base_model_out)

class ModelWrapper:
    def __init__(self):
        self.tokenizer = _find_tokenizer()
        self.preds = {}
        # default params
        vocab_size = len(self.tokenizer.vocab_list) if (self.tokenizer and hasattr(self.tokenizer, 'vocab_list')) else 2848
        self.cnn_model = get_cnn_model()
        self.encoder = TransformerEncoderBlock(embed_dim=768, dense_dim=2048, num_heads=4)
        self.decoder = TransformerDecoderBlock(embed_dim=768, ff_dim=2048, num_heads=4, vocab_size=vocab_size)
        self.model = ImageCaptioningModel(self.cnn_model, self.encoder, self.decoder)

    def to(self, device: str):
        return

    def _preprocess_image(self, path: str):
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (380, 380))
        img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
        return np.expand_dims(img, 0)

    def load_checkpoint(self, path: str):
        p = Path(path)
        import logging
        logger = logging.getLogger(__name__)

        # Infer vocab size
        tok_size = len(self.tokenizer) if self.tokenizer else 2848
        ckpt_vocab = None
        ckpt_prefix = None
        
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
                    if 'emb' in lname and 'position' not in lname and len(shape) >= 1:
                        ckpt_vocab = int(shape[0])
                        ckpt_prefix = cand
                        break
                if ckpt_vocab is not None:
                    logger.info(f"Inferred EfficientNetB4 checkpoint vocab size {ckpt_vocab} from {ckpt_prefix}")
                    break
            except Exception as e:
                logger.warning(f"Error checking variables in checkpoint candidate {cand}: {e}")
                continue

        vocab_size = ckpt_vocab if ckpt_vocab is not None else tok_size
        logger.info(f"Setting EfficientNetB4 decoder vocab_size to {vocab_size}")
        self.decoder = TransformerDecoderBlock(embed_dim=768, ff_dim=2048, num_heads=4, vocab_size=vocab_size)
        self.model = ImageCaptioningModel(self.cnn_model, self.encoder, self.decoder)

        prefix_to_load = ckpt_prefix or (str(p)[:-len('.index')] if p.suffix == '.index' else str(path))
        try:
            self.model.load_weights(prefix_to_load)
            logger.info(f"Successfully loaded weights using Keras load_weights from {prefix_to_load}")
            return
        except Exception as e:
            logger.warning(f"Keras load_weights failed for {prefix_to_load}: {e}")

        # If path is directory, search for predictions.json
        if p.is_dir():
            for name in ("predictions.json", "preds.json", "captions.json", "generated_captions.json"):
                f = p / name
                if f.exists():
                    try:
                        self.preds = json.loads(f.read_text(encoding='utf-8'))
                        return
                    except Exception:
                        continue

        if p.is_file() and p.suffix == '.json':
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    self.preds = data
                    return
            except Exception:
                pass

    def generate(self, img_id: Optional[str] = None, decode: str = "greedy", beam_size: int = 3, tokenizer=None):
        key = str(img_id)
        if key in self.preds:
            return {"caption": self.preds[key]}

        tok = tokenizer or self.tokenizer
        if tok is None:
            return {"caption": ""}

        # resolve image path
        img_path = Path(key)
        if not img_path.exists():
            candidates = [Path('/data'), Path('data'), Path('.')]
            found = None
            for root in candidates:
                try:
                    for p in root.rglob(img_path.name):
                        if p.is_file():
                            found = p
                            break
                    if found:
                        break
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

        img = self._preprocess_image(str(img_path))
        img_features = self.model.cnn_model(img)
        img_features = tf.expand_dims(img_features, 1) # Adding sequence dimension (1, 1, 768)
        encoded_img = self.model.encoder(img_features, training=False)

        decoded_caption = ["<start>"]
        attention_plot = []
        for _ in range(7): # SEQ_LENGTH - 1
            # Manual tokenization matching TextVectorization
            ids = [tok.word_index.get(w, 1) for w in decoded_caption]
            if len(ids) < 8:
                ids = ids + [0] * (8 - len(ids))
            else:
                ids = ids[:8]
            tokenized_caption = tf.constant([ids[:-1]], dtype=tf.int32)
            
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions, attention_scores = self.model.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask, return_attention=True
            )
            
            # Autoregressive prediction is at the index matching current sequence length - 1
            curr_pos = len(decoded_caption) - 1
            preds_at_pos = predictions[0, curr_pos, :]
            
            # Average attention weights across heads and extract for current step
            mean_attn = tf.reduce_mean(attention_scores, axis=1) # shape (batch_size, sequence_length, key_sequence_length)
            step_attn = mean_attn[0, curr_pos, :].numpy().tolist() # shape (key_sequence_length,)
            attention_plot.append(step_attn)
            
            # Find the best token index that is not in the blacklist
            sampled_token_indices = np.argsort(preds_at_pos.numpy())[::-1]
            sampled_token = None
            for token_idx in sampled_token_indices:
                w = tok.index_word.get(token_idx, '')
                if w not in ("[UNK]", "<pad>", "", "<start>"):
                    sampled_token = w
                    break
            
            if sampled_token is None or sampled_token == "<end>":
                break
                
            decoded_caption.append(sampled_token)

        caption = " ".join(decoded_caption[1:]).strip()
        return {"caption": caption, "attention": attention_plot}

def build_model() -> Tuple[ModelWrapper, Optional[object], str]:
    m = ModelWrapper()
    return m, m.tokenizer, 'cpu'
