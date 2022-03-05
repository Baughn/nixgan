import numpy as np
from PIL import Image
import jax
import time

import clip_jax

image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load('ViT-B/32', "cpu")

batch_size = 2048

devices = jax.local_devices()

print(f"jax devices: {devices}")

jax_params = jax.device_put_replicated(jax_params, devices)
image_fn = jax.pmap(image_fn)
text_fn = jax.pmap(text_fn)

jax_image = np.expand_dims(jax_preprocess(Image.open("CLIP.png")), (0, 1))
jax_image = np.repeat(jax_image, len(devices), axis=0)
jax_image = np.repeat(jax_image, batch_size, axis=1)

jax_text = np.expand_dims(clip_jax.tokenize(["a diagram"]), 0)
jax_text = np.repeat(jax_text, len(devices), axis=0)
jax_text = np.repeat(jax_text, batch_size, axis=1)

start = time.time()
jax_image_embed = image_fn(jax_params, jax_image)
jax_text_embed = text_fn(jax_params, jax_text)
total = time.time() - start
print(f"{total:.06}s to compile model")

start = time.time()
for i in range(16):
    jax_image_embed = np.array(image_fn(jax_params, jax_image))
    jax_text_embed = np.array(text_fn(jax_params, jax_text))

total = time.time() - start

print(f"{total:.06}s for 16 batches@bs={batch_size} per core")
print(f"{16*len(devices) * batch_size/total:.06} examples/s")

print("done!")