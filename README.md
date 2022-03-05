# CLIP (With Haiku + Jax!)

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

# Details

The ViT model and checkpoints have been ported to Haiku, while preserving the same output. See `tests/test_consistency.py` for details.

No JIT/pmap is performed, but pure inference functions for both the text and image encoders are provided from the the
`clip_jax.load()` function which should be easy to run/parallelize how you wish. See `test/tpu_bench.py` for an example of using pmap.

## Usage Example

```python
import numpy as np
from PIL import Image

import clip_jax

image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load('ViT-B/32', "cpu")

image = np.expand_dims(jax_preprocess(Image.open("CLIP.png")), 0)
text = clip_jax.tokenize(["a diagram", "a dog", "a cat"])

image_embed = image_fn(jax_params, image)
text_embed = text_fn(jax_params, text)
```

## TPU performance

On a TPU v3-8 with Jax tpu-vm alpha (`test/tpu_bench.py`):
```
10.1361s to compile model
43.9599s for 16 batches
5963.25 examples/s
```

## TODOs
- [x] Test on TPUs
- [ ] Easier control over precision and device placement
- [ ] Mixed precision training support
- [ ] Support RN50 model