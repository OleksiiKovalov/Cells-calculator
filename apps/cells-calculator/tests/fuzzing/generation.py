"""Image generators for the runtime fuzzer.

Every case is fully determined by (seed, index, profile, corpus), so any failure
is reproducible from its seed + index.
"""
import numpy as np

PROFILES = ("random", "edge", "corpus", "mixed")


def case_rng(seed, index):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(index) & 0xFFFFFFFF])


# Images are always (H, W, 3) uint8 — the contract read_img guarantees the model
# — so any failure is a real robustness bug, not an out-of-contract artifact.
def random_image(rng):
    h = int(rng.integers(1, 256))
    w = int(rng.integers(1, 256))
    kind = rng.choice(["noise", "solid", "gradient"])
    if kind == "noise":
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    if kind == "solid":
        return np.full((h, w, 3), int(rng.integers(0, 256)), dtype=np.uint8)
    row = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.repeat(np.tile(row, (h, 1))[:, :, None], 3, axis=2)
    return np.ascontiguousarray(img.astype(np.uint8))


def edge_image(rng):
    builders = [
        lambda: np.zeros((1, 1, 3), np.uint8),                                  # 1x1
        lambda: np.full((1, int(rng.integers(1, 512)), 3), 255, np.uint8),      # extreme aspect
        lambda: np.full((int(rng.integers(1, 512)), 1, 3), 255, np.uint8),
        lambda: np.zeros((int(rng.integers(1, 64)), int(rng.integers(1, 64)), 3), np.uint8),
        lambda: np.full((32, 32, 3), 255, np.uint8),                            # all white
        lambda: rng.integers(0, 256, (2, 2, 3), dtype=np.uint8),                # tiny noise
    ]
    return np.ascontiguousarray(builders[int(rng.integers(len(builders)))]())


def mutate_corpus(rng, path):
    import cv2
    from model.utils import read_img
    img = read_img(str(path))
    if img is None:
        return None
    ops = list(rng.permutation(["resize", "crop", "noise", "gray", "flip", "rot"]))
    for op in ops[: int(rng.integers(1, 4))]:
        if op == "resize":
            f = float(rng.uniform(0.2, 1.5))
            img = cv2.resize(img, (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f))))
        elif op == "crop":
            h, w = img.shape[:2]
            if h > 4 and w > 4:
                y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
                img = img[y0:y0 + max(1, h // 2), x0:x0 + max(1, w // 2)]
        elif op == "noise":
            n = rng.integers(0, 40, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        elif op == "gray" and img.ndim == 3:
            # Desaturate but keep 3 channels (read_img always returns RGB).
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        elif op == "flip":
            img = img[:, ::-1] if rng.random() < 0.5 else img[::-1]
        elif op == "rot":
            img = np.rot90(img, int(rng.integers(1, 4)))
    return np.ascontiguousarray(img)


def generate_case(seed, index, profile, corpus, corpus_prob):
    """Return (image, kind) for one fuzz case."""
    rng = case_rng(seed, index)
    if corpus and (profile == "corpus" or (profile == "mixed" and rng.random() < corpus_prob)):
        img = mutate_corpus(rng, corpus[int(rng.integers(len(corpus)))])
        if img is not None:
            return img, "corpus"
    if profile == "edge" or (profile == "mixed" and rng.random() < 0.3):
        return edge_image(rng), "edge"
    return random_image(rng), "random"
