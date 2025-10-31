# T5 Transfer Learning for Abstractive Summarization (TensorFlow)

This project demonstrates **transfer learning** with the **T5** (Text-To-Text Transfer Transformer) model for **abstractive text summarization** using the **CNN/DailyMail** dataset. We fine-tune a small pre-trained T5 checkpoint (`t5-small`) with **TensorFlow** and the Hugging Face **Transformers** and **Datasets** libraries, then generate summaries with beam search.

> Goal: Keep the code minimal while showing how to adapt a general-purpose pre-trained language model to a summarization task with a simple training loop and generation settings.

---

## Why Transfer Learning?
Pre-trained sequence-to-sequence models (like T5) have already learned general language patterns from large corpora. **Fine-tuning** them on a target dataset (CNN/DailyMail) helps the model:
- Learn task-specific behavior (here: summarization).
- Converge faster with fewer examples.
- Achieve better performance than training from scratch.

---

## Project Structure (What the script does)

1. **Load Pre-trained Assets**
   - `T5Tokenizer` and `TFT5ForConditionalGeneration` are loaded from `t5-small`.
2. **Load Dataset**
   - Uses Hugging Face `datasets.load_dataset("cnn_dailymail", "3.0.0")`.
   - Trains on a small split (e.g., `train[:10%]`) for speed.
3. **Preprocess**
   - Adds the T5 task prefix `"summarize:"` to each article.
   - Tokenizes inputs and targets to fixed lengths (inputs to 512 tokens, targets to 64 tokens).
4. **Create `tf.data.Dataset`**
   - Streams tokenized samples via a generator into a batched, prefetched pipeline.
5. **Compile and Train**
   - Compiles the model with Adam and `SparseCategoricalCrossentropy(from_logits=True)`.
   - Trains for a few epochs (e.g., 5) to demonstrate end-to-end fine-tuning.
6. **Generate Summaries**
   - Provides a helper function `generate_summary(text)` that encodes input and decodes model output.
   - Uses beam search (`num_beams=8`), `length_penalty=2.0`, `no_repeat_ngram_size=2`, and `early_stopping=True` for cleaner outputs.


## References
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*.
- Hugging Face Transformers: <https://huggingface.co/docs/transformers>
- Hugging Face Datasets: <https://huggingface.co/docs/datasets>

---

## License
This example is for educational purposes. Follow the licenses of the datasets and pre-trained models you use.
