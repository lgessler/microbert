from transformers import pipeline

nlp = pipeline('fill-mask', tokenizer="bert_out", model="bert_out")

s = "ⲁⲣ ϯⲉⲟⲟⲩ ⲙ [MASK] ⲛⲟⲩⲧⲉ ⲧⲱⲛⲉ ."

nlp(s)
