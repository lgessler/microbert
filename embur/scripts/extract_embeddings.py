import sys

from allennlp.models import load_archive

if __name__ == "__main__":
    archive = load_archive(sys.argv[1])
    vocab = archive.model.vocab
    weights = archive.model._backbone.embedder.token_embedder_tokens.state_dict()["weight"]

    with open(sys.argv[2], "w") as f:
        f.write(f"{weights.shape[0]} {weights.shape[1]}\n")
        for i in range(weights.shape[0]):
            token = vocab.get_token_from_index(i, "tokens")
            vec = weights[i].tolist()
            f.write(token + " " + " ".join(str(w) for w in vec) + "\n")
