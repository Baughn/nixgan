import sys, os
sys.path.append('./CLIP_JAX')
sys.path.append('./jax-guided-diffusion')
sys.path.append('./v-diffusion-jax')
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from PIL import Image, ImageOps
from braceexpand import braceexpand
from dataclasses import dataclass
from functools import partial
from subprocess import Popen, PIPE
import functools
import io
import math
import re
import requests
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init
from tqdm import tqdm

from lib.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from lib import util, openai

from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
import torch.utils.data
import torch

import diffusion as v_diffusion

from diffusion_models.common import DiffusionOutput, Partial, make_partial, blur_fft, norm1
from diffusion_models.cache import WeakCache
from diffusion_models.schedules import cosine, ddpm, ddpm2, spliced
from diffusion_models.perceptor import vit32, vit16, clip_size, normalize, get_vitl14
vitl14 = get_vitl14()

from diffusion_models.secondary import secondary1_wrap, secondary2_wrap
from diffusion_models.antijpeg import jpeg_wrap, jpeg_classifier_wrap
from diffusion_models.pixelart import pixelartv4_wrap, pixelartv6_wrap
from diffusion_models.pixelartv7 import pixelartv7_ic_wrap, pixelartv7_ic_attn_wrap
from diffusion_models.cc12m_1 import cc12m_1_wrap, cc12m_1_cfg_wrap
from diffusion_models.openai import make_openai_model, make_openai_finetune_model

import gc
import shutil
import imageio
from subprocess import Popen, PIPE
import os.path

outputFolderStatic = '/home/svein/web/GAN/jaxgan/'
cache_location = '/home/svein/.cache/jaxgan/'

def Loc(l):
  return os.path.join(cache_location, l)

model_location = Loc('models')


devices = jax.devices()
n_devices = len(devices)
print('Using device:', devices)

os.makedirs(model_location, exist_ok=True)

# Define necessary functions

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    local_path = os.path.join(model_location, basename)
    if os.path.exists(local_path):
        return local_path
    else:
        os.makedirs(f'{model_location}/tmp', exist_ok=True)
        Popen(['curl', url_or_path, '-o', f'{model_location}/tmp/{basename}']).wait()
        os.rename(f'{model_location}/tmp/{basename}', local_path)
        return local_path

# Implement lazy loading and caching of model parameters for all the different models.

gpu_cache = WeakCache(jnp.array)

def to_gpu(params):
  """Convert a pytree of params to jax, using cached arrays if they are still alive."""
  return jax.tree_util.tree_map(lambda x: gpu_cache(x) if type(x) is np.ndarray else x, params)

class LazyParams(object):
  """Lazily download parameters and load onto gpu. Parameters are kept in cpu memory and only loaded to gpu as long as needed."""
  def __init__(self, load):
    self.load = load
    self.params = None
  @staticmethod
  def pt(url, key=None):
    def load():
      params = jaxtorch.pt.load(fetch_model(url))
      if key is not None:
        return params[key]
      else:
        return params
    return LazyParams(load)
  def __call__(self):
    if self.params is None:
      self.params = jax.tree_util.tree_map(np.array, self.load())
    return to_gpu(self.params)


def grey(image):
    [*_, c, h, w] = image.shape
    return jnp.broadcast_to(image.mean(axis=-3, keepdims=True), image.shape)

def cutout_image(image, offsetx, offsety, size, output_size=224):
    """Computes (square) cutouts of an image given x and y offsets and size."""
    (c, h, w) = image.shape

    scale = jnp.stack([output_size / size, output_size / size])
    translation = jnp.stack([-offsety * output_size / size, -offsetx * output_size / size])
    return jax.image.scale_and_translate(image,
                                         shape=(c, output_size, output_size),
                                         spatial_dims=(1,2),
                                         scale=scale,
                                         translation=translation,
                                         method='lanczos3')

def cutouts_images(image, offsetx, offsety, size, output_size=224):
    f = partial(cutout_image, output_size=output_size)         # [c h w] [] [] [] -> [c h w]
    f = jax.vmap(f, in_axes=(0, None, None, None), out_axes=0) # [n c h w] [] [] [] -> [n c h w]
    f = jax.vmap(f, in_axes=(None, 0, 0, 0), out_axes=0)       # [n c h w] [k] [k] [k] -> [k n c h w]
    return f(image, offsetx, offsety, size)

@jax.tree_util.register_pytree_node_class
class MakeCutouts(object):
    def __init__(self, cut_size, cutn, cut_pow=1., p_grey=0.2, p_mixgrey=0.0):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.p_grey = p_grey
        self.p_mixgrey = p_mixgrey

    def __call__(self, input, key):
        [b, c, h, w] = input.shape
        rng = PRNG(key)
        max_size = min(h, w)
        min_size = min(h, w, self.cut_size)
        cut_us = jax.random.uniform(rng.split(), shape=[self.cutn//2])**self.cut_pow
        sizes = (min_size + cut_us * (max_size - min_size + 1)).astype(jnp.int32).clamp(min_size, max_size)
        offsets_x = jax.random.uniform(rng.split(), [self.cutn//2], minval=0, maxval=w - sizes)
        offsets_y = jax.random.uniform(rng.split(), [self.cutn//2], minval=0, maxval=h - sizes)
        cutouts = cutouts_images(input, offsets_x, offsets_y, sizes)

        B1 = 40
        B2 = 40
        lcut_us = jax.random.uniform(rng.split(), shape=[self.cutn//2])
        border = B1 + lcut_us * B2
        lsizes = (max(h,w) + border).astype(jnp.int32)
        loffsets_x = jax.random.uniform(rng.split(), [self.cutn//2], minval=w/2-lsizes/2-border, maxval=w/2-lsizes/2+border)
        loffsets_y = jax.random.uniform(rng.split(), [self.cutn//2], minval=h/2-lsizes/2-border, maxval=h/2-lsizes/2+border)
        lcutouts = cutouts_images(input, loffsets_x, loffsets_y, lsizes)

        cutouts = jnp.concatenate([cutouts, lcutouts], axis=0)

        greyed = grey(cutouts)

        grey_us = jax.random.uniform(rng.split(), shape=[self.cutn, b, 1, 1, 1])
        grey_rs = jax.random.uniform(rng.split(), shape=[self.cutn, b, 1, 1, 1])
        cutouts = jnp.where(grey_us < self.p_mixgrey, grey_rs * greyed + (1 - grey_rs) * cutouts, cutouts)

        grey_us = jax.random.uniform(rng.split(), shape=[self.cutn, b, 1, 1, 1])
        cutouts = jnp.where(grey_us < self.p_grey, greyed, cutouts)
        # Flip augmentation
        flip_us = jax.random.bernoulli(rng.split(), 0.5, [self.cutn, b, 1, 1, 1])
        cutouts = jnp.where(flip_us, jnp.flip(cutouts, axis=-1), cutouts)
        return cutouts

    def tree_flatten(self):
        return ([self.p_grey, self.cut_pow, self.p_mixgrey], (self.cut_size, self.cutn))

    @staticmethod
    def tree_unflatten(static, dynamic):
        (cut_size, cutn) = static
        (p_grey, cut_pow, p_mixgrey) = dynamic
        return MakeCutouts(cut_size, cutn, cut_pow, p_grey, p_mixgrey)

@jax.tree_util.register_pytree_node_class
class MakeCutoutsPixelated(object):
    def __init__(self, make_cutouts, factor=4):
        self.make_cutouts = make_cutouts
        self.factor = factor
        self.cutn = make_cutouts.cutn

    def __call__(self, input, key):
        [n, c, h, w] = input.shape
        input = jax.image.resize(input, [n, c, h*self.factor, w * self.factor], method='nearest')
        return self.make_cutouts(input, key)

    def tree_flatten(self):
        return ([self.make_cutouts], [self.factor])
    @staticmethod
    def tree_unflatten(static, dynamic):
        return MakeCutoutsPixelated(*dynamic, *static)

def spherical_dist_loss(x, y):
    x = norm1(x)
    y = norm1(y)
    return (x - y).square().sum(axis=-1).sqrt().div(2).arcsin().square().mul(2)

# Define combinators.

# These (ab)use the jax pytree registration system to define parameterised
# objects for doing various things, which are compatible with jax.jit.

# For jit compatibility an object needs to act as a pytree, which means implementing two methods:
#  - tree_flatten(self): returns two lists of the object's fields:
#       1. 'dynamic' parameters: things which can be jax tensors, or other pytrees
#       2. 'static' parameters: arbitrary python objects, will trigger recompilation when changed
#  - tree_unflatten(static, dynamic): reconstitutes the object from its parts

# With these tricks, you can simply define your cond_fn as an object, as is done
# below, and pass it into the jitted sample step as a regular argument. JAX will
# handle recompiling the jitted code whenever a control-flow affecting parameter
# is changed (such as cut_batches).

@jax.tree_util.register_pytree_node_class
class LerpModels(object):
    """Linear combination of diffusion models."""
    def __init__(self, models):
        self.models = models
    def __call__(self, x, t, key):
        outputs = [m(x,t,key) for (m,w) in self.models]
        v = sum(out.v * w for (out, (m,w)) in zip(outputs, self.models))
        pred = sum(out.pred * w for (out, (m,w)) in zip(outputs, self.models))
        eps = sum(out.eps * w for (out, (m,w)) in zip(outputs, self.models))
        return DiffusionOutput(v, pred, eps)
    def tree_flatten(self):
        return [self.models], []
    def tree_unflatten(static, dynamic):
        return LerpModels(*dynamic)

@jax.tree_util.register_pytree_node_class
class KatModel(object):
    def __init__(self, model, params, **kwargs):
      if isinstance(params, LazyParams):
        params = params()
      self.model = model
      self.params = params
      self.kwargs = kwargs
    @jax.jit
    def __call__(self, x, cosine_t, key):
        n = x.shape[0]
        alpha, sigma = cosine.to_alpha_sigma(cosine_t)
        v = self.model.apply(self.params, key, x, cosine_t.broadcast_to([n]), self.kwargs)
        pred = x * alpha - v * sigma
        eps = x * sigma + v * alpha
        return DiffusionOutput(v, pred, eps)
    def tree_flatten(self):
        return [self.params, self.kwargs], [self.model]
    def tree_unflatten(static, dynamic):
        [params, kwargs] = dynamic
        [model] = static
        return KatModel(model, params, **kwargs)

# A wrapper that causes the diffusion model to generate tileable images, by
# randomly shifting the image with wrap around.

def xyroll(x, shifts):
  return jax.vmap(partial(jnp.roll, axis=[1,2]), in_axes=(0, 0))(x, shifts)

@make_partial
def TilingModel(model, x, cosine_t, key):
  rng = PRNG(key)
  [n, c, h, w] = x.shape
  shift = jax.random.randint(rng.split(), [n, 2], -50, 50)
  x = xyroll(x, shift)
  out = model(x, cosine_t, rng.split())
  def unshift(val):
    return xyroll(val, -shift)
  return jax.tree_util.tree_map(unshift, out)

@make_partial
def PanoramaModel(model, x, cosine_t, key):
  rng = PRNG(key)
  [n, c, h, w] = x.shape
  shift = jax.random.randint(rng.split(), [n, 2], 0, [1, w])
  x = xyroll(x, shift)
  out = model(x, cosine_t, rng.split())
  def unshift(val):
    return xyroll(val, -shift)
  return jax.tree_util.tree_map(unshift, out)

@make_partial
def SymmetryModel(model, x, cosine_t, key):
  rng = PRNG(key)
  [n, c, h, w] = x.shape
  x = jnp.concatenate([x[:, :, :, :w//2], jnp.flip(x[:, :, :, :w//2],-1)], -1)
  out = model(x, cosine_t, rng.split())
  return out

# Secondary Model
secondary1_params = LazyParams.pt('https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet.pth')
secondary2_params = LazyParams.pt('https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth')

# Anti-JPEG model
jpeg_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/jpeg-db-oi-614.pt', key='params_ema')
jpeg_classifier_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/jpeg-classifier-72.pt', 'params_ema')

# Pixel art model
# There are many checkpoints supported with this model
pixelartv4_params = LazyParams.pt(
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v4_34.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v4_63.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v4_150.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v5_50.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v5_65.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v5_97.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v5_173.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_344.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_432.pt'
    'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_600.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_700.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_800.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_1000.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_2000.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-fgood_3000.pt'
    , key='params_ema'
)

pixelartv6_params = LazyParams.pt(
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-1000.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-2000.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-3000.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-4000.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-aug-900.pt'
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-aug-1300.pt'
    'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-aug-3000.pt'
    , key='params_ema'
)

pixelartv7_ic_params = LazyParams.pt(
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-ic-1400.pt'
    'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v7-large-ic-700.pt'
    , key='params_ema'
)

pixelartv7_ic_attn_params = LazyParams.pt(
    # 'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v6-ic-1400.pt'
    'https://set.zlkj.in/models/diffusion/pixelart/pixelart-v7-large-ic-attn-600.pt'
    , key='params_ema'
)

# Kat models

danbooru_128_model = v_diffusion.get_model('danbooru_128')
danbooru_128_params = LazyParams(lambda: v_diffusion.load_params(fetch_model('https://v-diffusion.s3.us-west-2.amazonaws.com/danbooru_128.pkl')))

wikiart_256_model = v_diffusion.get_model('wikiart_256')
wikiart_256_params = LazyParams(lambda: v_diffusion.load_params(fetch_model('https://v-diffusion.s3.us-west-2.amazonaws.com/wikiart_256.pkl')))

imagenet_128_model = v_diffusion.get_model('imagenet_128')
imagenet_128_params = LazyParams(lambda: v_diffusion.load_params(fetch_model('https://v-diffusion.s3.us-west-2.amazonaws.com/imagenet_128.pkl')))

# CC12M_1 model

cc12m_1_params = LazyParams.pt('https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth')
cc12m_1_cfg_params = LazyParams.pt('https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth')

# OpenAI models.

use_checkpoint = False # Set to True to save some memory

openai_512_model = openai.create_openai_512_model(use_checkpoint=use_checkpoint)
openai_512_params = openai_512_model.init_weights(jax.random.PRNGKey(0))
openai_512_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_finetune_008100.pt')
openai_512_wrap = make_openai_model(openai_512_model)

openai_256_model = openai.create_openai_256_model(use_checkpoint=use_checkpoint)
openai_256_params = openai_256_model.init_weights(jax.random.PRNGKey(0))
openai_256_params = LazyParams.pt('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt')
openai_256_wrap = make_openai_model(openai_256_model)

openai_512_finetune_wrap = make_openai_finetune_model(openai_512_model)
openai_512_finetune_params = LazyParams.pt('https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_openimages_epoch28_withfilter.pt')

# Aesthetic Model

def apply_partial(*args, **kwargs):
  def sub(f):
    return Partial(f, *args, **kwargs)
  return sub

aesthetic_model = nn.Linear(512, 10)
aesthetic_model.init_weights(jax.random.PRNGKey(0))
aesthetic_model_params = jaxtorch.pt.load(fetch_model('https://v-diffusion.s3.us-west-2.amazonaws.com/ava_vit_b_16_full.pth'))

def exec_aesthetic_model(params, embed):
  return jax.nn.log_softmax(aesthetic_model(Context(params, None), embed), axis=-1)
exec_aesthetic_model = Partial(exec_aesthetic_model, aesthetic_model_params)

# Losses and cond fn.

@make_partial
@apply_partial(exec_aesthetic_model)
def AestheticLoss(exec_aesthetic_model, target, scale, image_embeds):
    [k, n, d] = image_embeds.shape
    log_probs = exec_aesthetic_model(image_embeds)
    return -(scale * log_probs[:, :, target-1].mean(0)).sum()

@make_partial
@apply_partial(exec_aesthetic_model)
def AestheticExpected(exec_aesthetic_model, scale, image_embeds):
    [k, n, d] = image_embeds.shape
    probs = jax.nn.softmax(exec_aesthetic_model(image_embeds))
    expected = (probs * (1 + jnp.arange(10))).sum(-1)
    return -(scale * expected.mean(0)).sum()

@jax.tree_util.register_pytree_node_class
class CondCLIP(object):
    """Backward a loss function through clip."""
    def __init__(self, perceptor, make_cutouts, cut_batches, *losses):
        self.perceptor = perceptor
        self.make_cutouts = make_cutouts
        self.cut_batches = cut_batches
        self.losses = losses
    def __call__(self, x_in, key):
        n = x_in.shape[0]
        def main_clip_loss(x_in, key):
            cutouts = normalize(self.make_cutouts(x_in.add(1).div(2), key)).rearrange('k n c h w -> (k n) c h w')
            image_embeds = self.perceptor.embed_cutouts(cutouts)
            image_embeds = image_embeds.rearrange('(k n) c -> k n c', k=self.make_cutouts.cutn, n=n)
            return sum(loss_fn(image_embeds) for loss_fn in self.losses)
        num_cuts = self.cut_batches
        keys = jnp.stack(jax.random.split(key, num_cuts))
        main_clip_grad = jax.lax.scan(lambda total, key: (total + jax.grad(main_clip_loss)(x_in, key), key),
                                        jnp.zeros_like(x_in),
                                        keys)[0] / num_cuts
        return main_clip_grad
    def tree_flatten(self):
        return [self.perceptor, self.make_cutouts, self.losses], [self.cut_batches]
    @classmethod
    def tree_unflatten(cls, static, dynamic):
        [perceptor, make_cutouts, losses] = dynamic
        [cut_batches] = static
        return cls(perceptor, make_cutouts, cut_batches, *losses)

@make_partial
def SphericalDistLoss(text_embed, clip_guidance_scale, image_embeds):
    losses = spherical_dist_loss(image_embeds, text_embed).mean(0)
    return (clip_guidance_scale * losses).sum()

@make_partial
def InfoLOOB(text_embed, clip_guidance_scale, inv_tau, lm, image_embeds):
    all_image_embeds = norm1(image_embeds.mean(0))
    all_text_embeds = norm1(text_embed)
    sim_matrix = inv_tau * jnp.einsum('nc,mc->nm', all_image_embeds, all_text_embeds)
    xn = sim_matrix.shape[0]
    def loob(sim_matrix):
      diag = jnp.eye(xn) * sim_matrix
      off_diag = (1 - jnp.eye(xn))*sim_matrix + jnp.eye(xn) * float('-inf')
      return -diag.sum() + lm * jsp.special.logsumexp(off_diag, axis=-1).sum()
    losses = loob(sim_matrix) + loob(sim_matrix.transpose())
    return losses.sum() * clip_guidance_scale.mean() / inv_tau

@make_partial
def CondTV(tv_scale, x_in, key):
    def downscale2d(image, f):
        [c, n, h, w] = image.shape
        return jax.image.resize(image, [c, n, h//f, w//f], method='cubic')

    def tv_loss(input):
        """L2 total variation loss, as in Mahendran et al."""
        x_diff = input[..., :, 1:] - input[..., :, :-1]
        y_diff = input[..., 1:, :] - input[..., :-1, :]
        return x_diff.square().mean([1,2,3]) + y_diff.square().mean([1,2,3])

    def sum_tv_loss(x_in, f=None):
        if f is not None:
            x_in = downscale2d(x_in, f)
        return tv_loss(x_in).sum() * tv_scale
    tv_grad_512 = jax.grad(sum_tv_loss)(x_in)
    tv_grad_256 = jax.grad(partial(sum_tv_loss,f=2))(x_in)
    tv_grad_128 = jax.grad(partial(sum_tv_loss,f=4))(x_in)
    return tv_grad_512 + tv_grad_256 + tv_grad_128

@make_partial
def CondRange(range_scale, x_in, key):
    def loss(x_in):
        return jnp.abs(x_in - x_in.clamp(minval=-1,maxval=1)).mean()
    return range_scale * jax.grad(loss)(x_in)

@make_partial
def CondMSE(target, mse_scale, x_in, key):
    def mse_loss(x_in):
        return (x_in - target).square().mean()
    return mse_scale * jax.grad(mse_loss)(x_in)

@jax.tree_util.register_pytree_node_class
class MaskedMSE(object):
    # MSE loss. Targets the output towards an image.
    def __init__(self, target, mse_scale, mask, grey=False):
        self.target = target
        self.mse_scale = mse_scale
        self.mask = mask
        self.grey = grey
    def __call__(self, x_in, key):
        def mse_loss(x_in):
            if self.grey:
              return (self.mask * grey(x_in - self.target).square()).mean()
            else:
              return (self.mask * (x_in - self.target).square()).mean()
        return self.mse_scale * jax.grad(mse_loss)(x_in)
    def tree_flatten(self):
        return [self.target, self.mse_scale, self.mask], [self.grey]
    def tree_unflatten(static, dynamic):
        return MaskedMSE(*dynamic, *static)


@jax.tree_util.register_pytree_node_class
class MainCondFn(object):
    # Used to construct the main cond_fn. Accepts a diffusion model which will
    # be used for denoising, plus a list of 'conditions' which will
    # generate gradient of a loss wrt the denoised, to be summed together.
    def __init__(self, diffusion, conditions, blur_amount=None, use='pred'):
        self.diffusion = diffusion
        self.conditions = [c for c in conditions if c is not None]
        self.blur_amount = blur_amount
        self.use = use

    @jax.jit
    def __call__(self, key, x, cosine_t):
        rng = PRNG(key)
        n = x.shape[0]

        alphas, sigmas = cosine.to_alpha_sigma(cosine_t)

        def denoise(key, x):
            pred = self.diffusion(x, cosine_t, key).pred
            if self.use == 'pred':
                return pred
            elif self.use == 'x_in':
                return pred * sigmas + x * alphas
        (x_in, backward) = jax.vjp(partial(denoise, rng.split()), x)

        total = jnp.zeros_like(x_in)
        for cond in self.conditions:
            total += cond(x_in, rng.split())
        if self.blur_amount is not None:
          blur_radius = (self.blur_amount * sigmas / alphas).clamp(0.05,512)
          total = blur_fft(total, blur_radius.mean())
        final_grad = -backward(total)[0]

        # clamp gradients to a max of 0.2
        magnitude = final_grad.square().mean(axis=(1,2,3), keepdims=True).sqrt()
        # Change the two values after 'magnitude >' to set a max clamp. 
        final_grad = final_grad * jnp.where(magnitude > 0.2, 0.2 / magnitude, 1.0)
        return final_grad
    def tree_flatten(self):
        return [self.diffusion, self.conditions, self.blur_amount], [self.use]
    def tree_unflatten(static, dynamic):
        return MainCondFn(*dynamic, *static)


@jax.tree_util.register_pytree_node_class
class CondFns(object):
    def __init__(self, *conditions):
        self.conditions = conditions
    def __call__(self, key, x, t):
        rng = PRNG(key)
        total = jnp.zeros_like(x)
        for cond in self.conditions:
          total += cond(rng.split(), x, t)
        return total
    def tree_flatten(self):
        return [self.conditions], []
    def tree_unflatten(static, dynamic):
        [conditions] = dynamic
        return CondFns(*conditions)

def clamp_score(score):
  magnitude = score.square().mean(axis=(1,2,3), keepdims=True).sqrt()
  return score * jnp.where(magnitude > 0.1, 0.1 / magnitude, 1.0)


@make_partial
def BlurRangeLoss(scale, key, x, cosine_t):
    def blurred_pred(x, cosine_t):
      alpha, sigma = cosine.to_alpha_sigma(cosine_t)
      blur_radius = (sigma / alpha * 2).clamp(0.05,512)
      return blur_fft(x, blur_radius) / alpha.clamp(0.01)
    def loss(x):
        pred = blurred_pred(x, cosine_t)
        diff = pred - pred.clamp(minval=-1,maxval=1)
        return diff.square().sum()
    return clamp_score(-scale * jax.grad(loss)(x))

def sample_step(key, x, t1, t2, diffusion, cond_fn, eta):
    rng = PRNG(key)

    n = x.shape[0]
    alpha1, sigma1 = cosine.to_alpha_sigma(t1)
    alpha2, sigma2 = cosine.to_alpha_sigma(t2)

    # Run the model
    out = diffusion(x, t1, rng.split())
    eps = out.eps
    pred0 = out.pred

    # # Predict the denoised image
    # pred0 = (x - eps * sigma1) / alpha1

    # Adjust eps with conditioning gradient
    cond_score = cond_fn(rng.split(), x, t1)
    eps = eps - sigma1 * cond_score

    # Predict the denoised image with conditioning
    pred = (x - eps * sigma1) / alpha1

    # Negative eta allows more extreme levels of noise.
    ddpm_sigma = (sigma2**2 / sigma1**2).sqrt() * (1 - alpha1**2 / alpha2**2).sqrt()
    ddim_sigma = jnp.where(eta >= 0.0,
                           eta * ddpm_sigma, # Normal: eta interpolates between ddim and ddpm
                           -eta * sigma2)    # Extreme: eta interpolates between ddim and q_sample(pred)
    adjusted_sigma = (sigma2**2 - ddim_sigma**2).sqrt()

    # Recombine the predicted noise and predicted denoised image in the
    # correct proportions for the next step
    x = pred * alpha2 + eps * adjusted_sigma

    # Add the correct amount of fresh noise
    x += jax.random.normal(rng.split(), x.shape) * ddim_sigma
    return x, pred0

def process_prompt(clip, prompt):
  # Brace expansion might change later, not sure this is the best way to do it.
  expands = braceexpand(prompt)
  embeds = []
  for sub in expands:
    sub = sub.strip()
    mult = 1.0
    if '~' in sub:
      mult *= -1.0
    sub = sub.replace('~', '')
    if 'http' not in sub:
        embeds.append(mult * clip.embed_text(sub))
    else:
        init_pil = Image.open(fetch(sub))
        embeds.append(mult * clip.embed_image(init_pil))
  return norm1(sum(embeds))

def process_prompts(clip, prompts):
  return jnp.stack([process_prompt(clip, prompt) for prompt in prompts])

def expand(xs, batch_size):
  """Extend or truncate the list of prompts to the batch size."""
  return (xs * batch_size)[:batch_size]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(image_size, octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(image_size[1], image_size[0]), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(image_size[1], image_size[0]), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def get_perlin_init(perlin_mode, image_size):
    if perlin_mode == 'colour':
        init = create_perlin_noise(image_size, [1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise(image_size, [1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'grey':
        init = create_perlin_noise(image_size, [1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise(image_size, [1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise(image_size, [1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise(image_size, [1.5**-i*0.5 for i in range(8)], 4, 4, True)
    init_array = jnp.array(TF.to_tensor(init).add(TF.to_tensor(init2))).unsqueeze(0).mul(2).sub(1)
    del init, init2
    return init_array

def setres(resSize, imageWidth, imageHeight):
    if resSize != "Custom":
        if resSize == "SmallSquare":
            image_size = (256, 256)
        elif resSize == "Square":
            image_size = (512, 512)
        elif resSize == "LargeSquare":
            image_size = (768, 768)
        elif resSize == "SmallLandscape":
            image_size = (448, 256)
        elif resSize == "Landscape1":
            image_size = (896, 512)
        elif resSize == "Landscape2":
            image_size = (1024, 576)
        elif resSize == "LargeLandscape":
            image_size = (1280, 704)
    else:
        image_size = (round(imageWidth/64)*64, round(imageHeight/64)*64)
    return image_size

# Change outputFolder depending on diffusion model
def get_output_folder(outputFolder, choose_diffusion_model, batch_outputFolder, use_batch_outputFolder):
    yearMonth = time.strftime('/%Y-%m/')
    outputFolder = outputFolderStatic+v2+choose_diffusion_model+yearMonth

    if use_batch_outputFolder and not batch_outputFolder == "":
        outputFolder += "batches/"+batch_outputFolder+"/"
    os.makedirs(outputFolder, exist_ok=True)
    
    return outputFolder

def get_local_ouput_folder(choose_diffusion_model, batch_outputFolder, use_batch_outputFolder):
    localOutputFolder = "samples/"+choose_diffusion_model+"/"

    if use_batch_outputFolder and not batch_outputFolder == "":
        localOutputFolder += "batches/"+batch_outputFolder+"/"

    return localOutputFolder

def get_save_every(steps):
    # 200 is best for splitting up total frames for showing changes without being too fast
    saveEvery = steps//200 if steps > 250 else 1
    secondsOfVideo = round(0.064 * steps + 6 + -0.024 * steps) if steps < 250 else 16
    return saveEvery, secondsOfVideo

class LerpWeightError(Exception):
       pass

class CombinationError(Exception):
       pass

# Kat's resize and crop method for init images
def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])

def build_InfoLOOB_prompt(prompt_template, items, use_InfoLOOB_prompt):
    items = [x.strip() for x in items.split('|')]
    temp = prompt_template.format(item=','.join(items))
    all_title = [prompt_template.format(item=item) for item in items]

    newline = "\n"
    splitprompts = f"{newline}{newline.join(all_title)}"
    print("InfoLOOB prompts:", splitprompts) if use_InfoLOOB_prompt else None
    return all_title

def cleanup_image_prompt(prompt):
    prompt = prompt.replace(".png ",".png").replace(".jpg ",".jpg").replace(" http","http")
    link = prompt.replace("#pixelart","").strip()
    prompt = '{'+link+',#pixelart}' if '#pixelart' in prompt and (".png" or ".jpg") in prompt else prompt
    return prompt

# Throw all the garbage in here so it doesn't clog up settings
def qol_setup(resSize, all_title, init_image, seed, choose_diffusion_model, imageWidth, imageHeight, batch_outputFolder, use_batch_outputFolder, use_InfoLOOB_prompt):
  
    choose_diffusion_model, sep, tail = choose_diffusion_model.partition(' |')
    outputFolder = outputFolderStatic

    resSize, sep, tail = resSize.partition(' |')
    image_size = setres(resSize, imageWidth, imageHeight)

    if type(all_title) == str:
        title = expand([x.strip() + " #pixelart" for x in all_title.split('||')] if "PixelArt" in choose_diffusion_model else [x.strip() for x in all_title.split('||')], batch_size)
    else:
        title = expand([x.strip() + " #pixelart" for x in all_title] if "PixelArt" in choose_diffusion_model else [x.strip() for x in all_title], batch_size)
    title = [cleanup_image_prompt(prompt) for prompt in title]

    matches = 0
    match = title[0]
    for prompt in title:
        matches += 1 if prompt == match else 0    
    splitprompt = 0 if matches == batch_size else 1

    if init_image == '':
        init_image = None
    if seed == '':
        seed = None
    
    def load_image(url):
        init_array = Image.open(fetch(url)).convert('RGB')
        init_array = resize_and_center_crop(init_array, image_size)
        init_array = jnp.array(TF.to_tensor(init_array)).unsqueeze(0).mul(2).sub(1)
        return init_array
    if type(init_image) is str:
        init_array = jnp.concatenate([load_image(it) for it in braceexpand(init_image)], axis=0)
    else:
        init_array = get_perlin_init('colour', image_size)

    schedule = jnp.linspace(starting_noise, 0, steps+1)
    schedule = spliced.to_cosine(schedule)

    return image_size, splitprompt, title, init_image, seed, outputFolder, choose_diffusion_model, init_array, schedule


# Advanced Settings #

#@markdown **CLIP Perceptor Selection:** *aesthetic_scale has no effect if vitb16 is false. vitl14 is slower and vram intensive.*
use_vitb16 = True #@param {type:"boolean"}
use_vitb32 = True #@param {type:"boolean"}
use_vitl14 = False #@param {type:"boolean"}

#@markdown ---

#@markdown ##### **Model modifications:** *Only one can be used at a time.*
#@markdown Produces a tiling effect for the image so the edges of itself wrap around.
use_tiled_model = False #@param {type:"boolean"}
#@markdown Moves the perceptor to give a 360 view of the scene.
use_panorama = False #@param {type:"boolean"}
#@markdown Enforces L/R symmetry during the diffusion process.
use_symmetry_model = False #@param {type:"boolean"}

if use_tiled_model and use_panorama and use_symmetry_model:
    raise CombinationError("Only one model modification is allowed at a single time.")

#@markdown ---

#@markdown InfoLOOB is an alternate way of using loss to guide the image to the text prompt when using a `batch_size` of greater than 1, uses `lm` to adjust similarity.
use_InfoLOOB = False #@param {type:"boolean"}

#@markdown Used with InfoLOOB when `batch_size` is 2 or greater, higher lm makes each image in the batch attracted to its own prompt and repelled from the other prompts. 
lm = 0.3 #@param {type:"raw"} 
inv_tau = 10000 #@param {type:"integer"} 

#@markdown

#@markdown **InfoLOOB prompt builder:** *Used if use_InfoLOOB_prompt is True.*
use_InfoLOOB_prompt = False #@param {type:"boolean"}
#@markdown ##### Insert `{item}` anywhere in prompt_template to have varied subjects or styles for infoloob to use. items are separated with `|`.
prompt_template = "\"A scenic view of the blue mountains\", {item}" #@param {type:"string"}
items = "matte painting trending on artstation | oil painting trending on artstation | digital art matte painting | digital art oil painting" #@param {type:"string"}
all_title = build_InfoLOOB_prompt(prompt_template, items, use_InfoLOOB_prompt)

#@markdown ---

#@markdown 0.0: DDIM | 1.0: DDPM | -1.0: Extreme noise, if a continuous number is used this will act as ratio and will modulate accordingly. 
eta = 1.0 #@param {type:"number"}

#@markdown ---

#@markdown A human aesthetics rating used to guide generation. Scales between -16.0 (Bad) to 16.0 (Perfect), if set to zero the aesthetics model won't be used. Works best with OpenAIFinetune and cc12m models.
aesthetics_scale = 6.0 #@param
aesthetics_scale = float(aesthetics_scale)

#@markdown ---
#@markdown #### **Init image:**  

#@markdown This can be a local path to the image file uploaded to the runtime (`init1.png`), or a URL to the image.
init_image = '' #@param {type:"string"}
#@markdown `starting_noise` (default: 1.0) The start of generation will look more like the `init_image` if it is used, generally 0.5-0.8 is good. The overnoising limit is 1.5.
starting_noise = 1.0 #@param {type:"raw"}
#@markdown MSE loss between the output and the `init_image`, guides the result look more like the init_image` (should be between 0 and width\*height*3). 
init_weight_mse = 0 #@param {type:"raw"}  

#@markdown `ic_cond` is used with PixelArtv7, this acts as a sort of init image for the model.
ic_cond = "https://irc.zlkj.in/uploads/eebeaf1803e898ac/88552154_p0%20-%20Coral.png" #@param {type:"string"}
# https://irc.zlkj.in/uploads/eebeaf1803e898ac/88552154_p0%20-%20Coral.png


#@title Model Alchemy
#@markdown Combines the outputs of different models, used if LerpedModels is chosen as the diffusion model.
#@markdown The `cond_model` is a secondary model used to help diffuse, `secondary2` is best for speed.
choose_cond_model = "secondary2" #@param ["secondary2", "OpenAI256", "PixelArtv6", "PixelArtv7", "PixelArtv4", "cc12m", "cc12m_cfg", "WikiArt", "Danbooru", "Imagenet128"]
lerpWeights = []

#@markdown ---
#@markdown The total sum of weights must add up to 1.0.
#@markdown ##### `use_antijpeg` will include the antijpeg model in the lerp, resulting in clearer results. `use_MakeCutoutsPixelated` will use the cutout method meant for the pixelart models.
use_antijpeg = True #@param {type:"boolean"}
use_MakeCutoutsPixelated = False #@param {type:"boolean"}

OpenAI512_weight = 0 #@param {type:"number"}
if OpenAI512_weight != 0:
    lerpWeights.append(OpenAI512_weight)

OpenAI256_weight = 0 #@param {type:"number"}
if OpenAI256_weight != 0:
    lerpWeights.append(OpenAI256_weight)

OpenAIFinetune_weight = 0.7 #@param {type:"number"}
if OpenAIFinetune_weight != 0:
    lerpWeights.append(OpenAIFinetune_weight)

PixelArtv4_weight = 0 #@param {type:"number"}
if PixelArtv4_weight != 0:
    lerpWeights.append(PixelArtv4_weight)

PixelArtv6_weight = 0 #@param {type:"number"}
if PixelArtv6_weight != 0:
    lerpWeights.append(PixelArtv6_weight)

PixelArtv7_weight =  0#@param {type:"number"}
if PixelArtv7_weight != 0:
    lerpWeights.append(PixelArtv7_weight)

cc12m_weight = 0 #@param {type:"number"}
if cc12m_weight != 0:
    lerpWeights.append(cc12m_weight)

cc12m_cfg_weight = 0 #@param {type:"number"}
if cc12m_cfg_weight != 0:
    lerpWeights.append(cc12m_cfg_weight)

WikiArt_weight = 0.3 #@param {type:"number"}
if WikiArt_weight != 0:
    lerpWeights.append(WikiArt_weight)

Danbooru_weight = 0 #@param {type:"number"}
if Danbooru_weight != 0:
    lerpWeights.append(Danbooru_weight)

Imagenet128_weight = 0 #@param {type:"number"}
if Imagenet128_weight != 0:
    lerpWeights.append(Imagenet128_weight)

secondary2_weight = 0 #@param {type:"number"}
if secondary2_weight != 0:
    lerpWeights.append(secondary2_weight)

totalWeight = sum(lerpWeights)
if totalWeight != 1.0:
    raise LerpWeightError("Total weights must add up to 1.0.")


#@markdown `all_title` - Your text prompt. ('#pixel art' is appended to the end of the prompt if using PixelArt)

# When batch_size is greater than 1, you can use '||' to seperate the prompts 
# and use different prompts for each image in the batch. 

if not use_InfoLOOB_prompt:
    all_title = sys.argv[1]
steps =  250 #@param {type:"raw"}
n_batches = 2 #@param {type:"integer"}        
batch_size =  1#@param {type:"integer"}
#@markdown Select the diffusion model: *Resizing the smaller models is very vram intensive and leads to some quality loss. Try generating at the native resolution and using the result as an init.*
choose_diffusion_model = "OpenAIFinetune | x512" #@param ["OpenAI | x512", "OpenAIFinetune | x512", "LerpedModels | lerp settings are in the Model Alchemy dropdown", "cc12m | x256 (CLIP conditioned)", "cc12m_cfg | x256 (CLIP conditioned, CLIP free guidance)", "PixelArtv4 | x256", "WikiArt | x256", "PixelArtv7_ic_attn | x128 (Instance cross attention)", "PixelArtv6 | x128","Danbooru | x128"]

#@markdown ---
#@markdown ### Image dimensions  
#@markdown Set to "Custom" to enter a custom resolution below.
resSize = "LargeLandscape | 1280x704" #@param ["Custom", "SmallSquare | 256x256", "Square | 512x512", "LargeSquare | 768x768", "SmallLandscape | 448x256", "Landscape1 | 896x512", "Landscape2 | 1024x576", "LargeLandscape | 1280x704"]
#@markdown Custom resolution - A resSize other than "Custom" overrides these values.
imageWidth = 0 #@param {type:"integer"}
imageHeight = 0#@param {type:"integer"}

#@markdown ---
#@markdown ### cutn settings 
#@markdown The effective value of cutn is cutn * cut_batches.
cutn =  32#@param {type:"integer"}
cut_batches =  4 #@param {type:"integer"}
#@markdown Affects the size of cutouts. Larger cut_pow -> smaller cutouts (down to the min of 224x244)
cut_pow = 1 #@param {type:"raw"}
make_cutouts = MakeCutouts(clip_size, cutn, cut_pow=cut_pow, p_mixgrey=0.0)
# mixgrey does a lerp between coloured and greyscale which is supposed to make the colours brighter

#@markdown ---
#@markdown ## Advanced settings
clip_guidance_scale = 4000 #@param {type:"integer"}
clip_guidance_scale = jnp.array([clip_guidance_scale]*batch_size)
cfg_guidance_scale = 6.0 #@param {type:"number"}    
tv_scale = 2000 #@param {type:"integer"}
range_scale = 1000 #@param {type:"integer"}

antijpeg_openai_models = True #@param {type: "boolean"}
antijpeg_guidance_scale =  10000 #@param {type:"integer"} 
#@markdown if `seed` is blank, the current Unix timestamp (in seconds) will be used. This should be a number.
seed = '' #@param {type:"raw"}

#@markdown ---
#@markdown Save an intermediate result during generation at the same rate as displayRate.
save_intermediate_frames = True #@param {type:"boolean"}

#@markdown Save your results into a subfolder for more organisation.
batch_outputFolder = "" #@param {type:"string"}
use_batch_outputFolder = False #@param {type:"boolean"}

#QoL Stuff function
image_size, splitprompt, title, init_image, seed, outputFolder, choose_diffusion_model, init_array, schedule = qol_setup(resSize, all_title, init_image, seed, choose_diffusion_model, imageWidth, imageHeight, batch_outputFolder, use_batch_outputFolder, use_InfoLOOB_prompt)


def config():
    # Configure models and load parameters onto gpu.
    # We do this in a function to avoid leaking gpu memory.
    print("Loading", choose_diffusion_model+"...")

    if choose_diffusion_model == "LerpedModels":
        # -- Combine different models to a single output --
        
        modelsToLerp = []
        cond_model = None

        if OpenAI512_weight != 0:
            openai512Lerp = openai_512_wrap(openai_512_params())
            modelsToLerp.append(openai512Lerp)
        if OpenAI256_weight != 0 or choose_cond_model == "OpenAI256":
            openai256Lerp = openai_256_wrap(openai_256_params())
            modelsToLerp.append(openai256Lerp) if OpenAI256_weight != 0 else None
            cond_model = openai256Lerp if choose_cond_model == "OpenAI256" else cond_model
        if OpenAIFinetune_weight != 0:
            openaifinetuneLerp = openai_512_finetune_wrap(openai_512_finetune_params())
            modelsToLerp.append(openaifinetuneLerp)
        if PixelArtv4_weight != 0 or choose_cond_model == "PixelArtv4":
            pixelartv4Lerp = pixelartv4_wrap(pixelartv4_params())
            modelsToLerp.append(pixelartv4Lerp) if PixelArtv4_weight != 0 else None
            cond_model = pixelartv4Lerp if choose_cond_model == "PixelArtv4" else cond_model
        if PixelArtv6_weight != 0 or choose_cond_model == "PixelArtv6":
            pixelartv6Lerp = pixelartv6_wrap(pixelartv6_params())
            modelsToLerp.append(pixelartv6Lerp) if PixelArtv6_weight != 0 else None
            cond_model = pixelartv6Lerp if choose_cond_model == "PixelArtv6" else cond_model
        if PixelArtv7_weight != 0 or choose_cond_model == "PixelArtv7":
            cond = jnp.array(TF.to_tensor(Image.open(fetch(ic_cond)).convert('RGB').resize(image_size,Image.BICUBIC))) * 2 - 1
            cond = cond.broadcast_to([batch_size, 3, image_size[1], image_size[0]])
            pixelartv7Lerp = pixelartv7_ic_attn_wrap(pixelartv7_ic_attn_params(), cond=cond, cfg_guidance_scale=cfg_guidance_scale)
            modelsToLerp.append(pixelartv7Lerp) if PixelArtv7_weight != 0 else None
            cond_model = pixelartv7Lerp if choose_cond_model == "PixelArtv7" else cond_model
        if cc12m_weight != 0 or choose_cond_model == "cc12m":
            cc12mLerp = cc12m_1_wrap(cc12m_1_params(), clip_embed=process_prompts(vit16, title))
            modelsToLerp.append(cc12mLerp) if cc12m_weight != 0 else None
            cond_model = cc12mLerp if choose_cond_model == "cc12m" else cond_model
        if cc12m_cfg_weight != 0 or choose_cond_model == "cc12m_cfg":
            cc12m_cfgLerp = cc12m_1_cfg_wrap(cc12m_1_cfg_params(), clip_embed=process_prompts(vit16, title), cfg_guidance_scale=cfg_guidance_scale)
            modelsToLerp.append(cc12m_cfgLerp) if cc12m_cfg_weight != 0 else None
            cond_model = cc12m_cfgLerp if choose_cond_model == "cc12m_cfg" else cond_model
        if WikiArt_weight != 0 or choose_cond_model == "WikiArt":
            wikiartLerp = KatModel(wikiart_256_model, wikiart_256_params())
            modelsToLerp.append(wikiartLerp) if WikiArt_weight != 0 else None
            cond_model = wikiartLerp if choose_cond_model == "WikiArt" else cond_model
        if Danbooru_weight != 0 or choose_cond_model == "Danbooru":
            danbooruLerp = KatModel(danbooru_128_model, danbooru_128_params())
            modelsToLerp.append(danbooruLerp) if Danbooru_weight != 0 else None
            cond_model = danbooruLerp if choose_cond_model == "Danbooru" else cond_model
        if Imagenet128_weight != 0 or choose_cond_model == "Imagenet128":
            Imagenet128Lerp = diffusion = KatModel(imagenet_128_model, imagenet_128_params())
            modelsToLerp.append(Imagenet128Lerp) if Imagenet128_weight != 0 else None
            cond_model = Imagenet128Lerp if choose_cond_model == "Imagenet128" else cond_model
        if secondary2_weight != 0 or choose_cond_model == "secondary2":
            secondary2 = secondary2_wrap(secondary2_params())
            modelsToLerp.append(secondary2) if secondary2_weight != 0 else None
            cond_model = secondary2 if choose_cond_model == "secondary2" else cond_model

        if use_antijpeg:
            jpeg_0 = jpeg_wrap(jpeg_params(), cond=jnp.array([0]*batch_size)) # Clean class
            jpeg_1 = jpeg_wrap(jpeg_params(), cond=jnp.array([2]*batch_size)) # Unconditional class
            modelsToLerp.append(jpeg_0)
            lerpWeights.append(1.0)
            modelsToLerp.append(jpeg_1)
            lerpWeights.append(-1.0)
            jpeg_classifier_fn = jpeg_classifier_wrap(jpeg_classifier_params(),
                                                      guidance_scale=antijpeg_guidance_scale, # will generally depend on image size
                                                      flood_level=0.7, # Prevent over-optimization
                                                      blur_size=3.0)

            cond_fn = CondFns(MainCondFn(cond_model, [
                      CondCLIP(vitl14, make_cutouts if not use_MakeCutoutsPixelated else MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                      CondCLIP(vit32, make_cutouts if not use_MakeCutoutsPixelated else MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                      CondCLIP(vit16, make_cutouts if not use_MakeCutoutsPixelated else MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm),
                      AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                      CondTV(tv_scale) if tv_scale > 0 else None,
                      CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                      CondRange(range_scale) if range_scale > 0 else None,
                      ]), jpeg_classifier_fn)
        else:
            cond_fn = CondFns(MainCondFn(cond_model, [
                      CondCLIP(vitl14, make_cutouts if not use_MakeCutoutsPixelated else MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                      CondCLIP(vit32, make_cutouts if not use_MakeCutoutsPixelated else MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                      CondCLIP(vit16, make_cutouts if not use_MakeCutoutsPixelated else MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm),
                      AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                      CondTV(tv_scale) if tv_scale > 0 else None,
                      CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                      CondRange(range_scale) if range_scale > 0 else None,
                      ]))
            
        diffusion = LerpModels([(model, weight) for model, weight in zip(modelsToLerp, lerpWeights)])

    elif choose_diffusion_model == 'OpenAI':
        openai = openai_512_wrap(openai_512_params())
        secondary2 = secondary2_wrap(secondary2_params())
        cond_model = secondary2

        if antijpeg_openai_models:
            jpeg_0 = jpeg_wrap(jpeg_params(), cond=jnp.array([0]*batch_size)) # Clean class
            jpeg_1 = jpeg_wrap(jpeg_params(), cond=jnp.array([2]*batch_size)) # Unconditional class
            jpeg_classifier_fn = jpeg_classifier_wrap(jpeg_classifier_params(),
                                                      guidance_scale=antijpeg_guidance_scale, # will generally depend on image size
                                                      flood_level=0.7, # Prevent over-optimization
                                                      blur_size=3.0)

            diffusion = LerpModels([(openai, 1.0),
                                    (jpeg_0, 1.0),
                                    (jpeg_1, -1.0)])
            
            cond_fn = CondFns(MainCondFn(cond_model, [
                      CondCLIP(vitl14, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                      CondCLIP(vit32, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                      CondCLIP(vit16, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm),
                      AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                      CondTV(tv_scale) if tv_scale > 0 else None,
                      CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                      CondRange(range_scale) if range_scale > 0 else None,
                      ]), jpeg_classifier_fn)
        else:
            diffusion = openai

            cond_fn = CondFns(MainCondFn(cond_model, [
                      CondCLIP(vitl14, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                      CondCLIP(vit32, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                      CondCLIP(vit16, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm),
                      AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                      CondTV(tv_scale) if tv_scale > 0 else None,
                      CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                      CondRange(range_scale) if range_scale > 0 else None,
                      ]))

    elif choose_diffusion_model in ('WikiArt', 'Danbooru'):
        if choose_diffusion_model == 'WikiArt':
            diffusion = KatModel(wikiart_256_model, wikiart_256_params())
        elif choose_diffusion_model == 'Danbooru':
            diffusion = KatModel(danbooru_128_model, danbooru_128_params())
        cond_model = diffusion
      
        cond_fn = MainCondFn(cond_model, [
                      CondCLIP(vitl14, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                      CondCLIP(vit32, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                      CondCLIP(vit16, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm),
                      AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                      CondTV(tv_scale) if tv_scale > 0 else None,
                      CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                      CondRange(range_scale) if range_scale > 0 else None,
                      ])

    elif 'PixelArt' in choose_diffusion_model:
        if choose_diffusion_model == 'PixelArtv7_ic_attn':
            # -- pixel art model --
            cond = jnp.array(TF.to_tensor(Image.open(fetch(ic_cond)).convert('RGB').resize(image_size,Image.BICUBIC))) * 2 - 1
            cond = cond.broadcast_to([batch_size, 3, image_size[1], image_size[0]])
            diffusion = pixelartv7_ic_attn_wrap(pixelartv7_ic_attn_params(), cond=cond, cfg_guidance_scale=cfg_guidance_scale)
        elif choose_diffusion_model == 'PixelArtv6':
            diffusion = pixelartv6_wrap(pixelartv6_params())
        elif choose_diffusion_model == 'PixelArtv4':
            diffusion = pixelartv4_wrap(pixelartv4_params())
        cond_model = diffusion

        cond_fn = MainCondFn(cond_model, [
                      CondCLIP(vitl14, MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                      CondCLIP(vit32, MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                      CondCLIP(vit16, MakeCutoutsPixelated(make_cutouts), cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit16, title), clip_guidance_scale, inv_tau, lm),
                      AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                      CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                      ])

    elif choose_diffusion_model == "cc12m":
        diffusion = cc12m_1_wrap(cc12m_1_params(), clip_embed = process_prompts(vit16,title).squeeze(0) if ('.png' or '.jpg') in all_title else process_prompts(vit16,title))
        cond_model = diffusion

        cond_fn = MainCondFn(cond_model, [
                        CondCLIP(vitl14, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                        CondCLIP(vit32, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                        CondCLIP(vit16, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit16, title), clip_guidance_scale, inv_tau, lm),
                        AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                        CondTV(tv_scale) if tv_scale > 0 else None,
                        CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                        CondRange(range_scale) if range_scale > 0 else None,
                        ])

        
    elif choose_diffusion_model == 'cc12m_cfg':
        diffusion = cc12m_1_cfg_wrap(cc12m_1_cfg_params(), clip_embed = process_prompts(vit16,title).squeeze(0) if ('.png' or '.jpg') in all_title else process_prompts(vit16,title), cfg_guidance_scale=cfg_guidance_scale)
        cond_fn = CondFns()
        
    elif choose_diffusion_model == 'OpenAIFinetune':
        openaifinetune = openai_512_finetune_wrap(openai_512_finetune_params())
        cond_model = secondary2_wrap(secondary2_params())

        if antijpeg_openai_models:
            jpeg_0 = jpeg_wrap(jpeg_params(), cond=jnp.array([0]*batch_size)) # Clean class
            jpeg_1 = jpeg_wrap(jpeg_params(), cond=jnp.array([2]*batch_size)) # Unconditional class
            jpeg_classifier_fn = jpeg_classifier_wrap(jpeg_classifier_params(),
                                                      guidance_scale=antijpeg_guidance_scale, # will generally depend on image size
                                                      flood_level=0.7, # Prevent over-optimization
                                                      blur_size=3.0)
            
            diffusion = LerpModels([(openaifinetune, 1.0),
                                    (jpeg_0, 1.0),
                                    (jpeg_1, -1.0)])
            
            
            cond_fn = CondFns(MainCondFn(cond_model, [ 
                        CondCLIP(vitl14, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                        CondCLIP(vit32, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                        CondCLIP(vit16, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit16, title), clip_guidance_scale, inv_tau, lm), 
                        AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                        CondTV(tv_scale) if tv_scale > 0 else None,
                        CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                        CondRange(range_scale) if range_scale > 0 else None,
                        ]), jpeg_classifier_fn)
    
        else:
            diffusion = openaifinetune
            cond_fn = MainCondFn(cond_model, [
                        CondCLIP(vitl14, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vitl14, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vitl14, title), clip_guidance_scale, inv_tau, lm)) if use_vitl14 else None,
                        CondCLIP(vit32, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit32, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit32, title), clip_guidance_scale, inv_tau, lm)) if use_vitb32 else None,
                        CondCLIP(vit16, make_cutouts, cut_batches, SphericalDistLoss(process_prompts(vit16, title), clip_guidance_scale) if not use_InfoLOOB else InfoLOOB(process_prompts(vit16, title), clip_guidance_scale, inv_tau, lm),
                        AestheticExpected(jnp.array([aesthetics_scale,aesthetics_scale,aesthetics_scale,aesthetics_scale]))) if use_vitb16 else None,
                        CondTV(tv_scale) if tv_scale > 0 else None,
                        CondMSE(init_array, init_weight_mse) if init_weight_mse > 0 else None,
                        CondRange(range_scale) if range_scale > 0 else None,
                        ])
    if use_tiled_model:
        diffusion = TilingModel(diffusion)
    if use_symmetry_model:
        diffusion = SymmetryModel(diffusion)
    return diffusion, cond_fn

diffusion, cond_fn = config()
print("Using", choose_diffusion_model+".")


#@title Diffuse!
displayRate = 25 #@param {type:"integer"}
#@markdown ---
saveVideo = False #@param {type:"boolean"}


def sanitize(title):
  return title[:200].replace('/', '_').replace('.', '_').replace(' ', '_')

@torch.no_grad()
def run():
    localOutputFolder = get_local_ouput_folder(choose_diffusion_model, batch_outputFolder, use_batch_outputFolder)
    saveEvery, secondsOfVideo = get_save_every(steps)

    nColumns = math.ceil(math.sqrt(batch_size))
    if batch_size == 7 or batch_size == 8:
        nColumns = 4
    batchNColumns = math.ceil(math.sqrt(n_batches  * batch_size)) 
    videoBatchNColumns = math.ceil(math.sqrt(n_batches))
    if n_batches * batch_size == 7 or n_batches * batch_size == 8:
        batchNColumns = 4

    if seed is None:
        local_seed = int(time.time())
    else:
        local_seed = seed
    rng = PRNG(jax.random.PRNGKey(local_seed))

    newline = "\n"
    splitprompts = f"run of: {newline}{newline.join(title)} {newline}with seed {local_seed}..."
    print(f'Starting {splitprompts}' if splitprompt > 0 else f'Starting run of ({title[0]}) with seed {local_seed}...')

    timestring = time.strftime('%Y%m%d%H%M%S')
    for i in range(n_batches):
        frameIteration = 0
        hmTimeStringFolder = time.strftime('%H-%M')
        
        ts = schedule
        alphas, sigmas = cosine.to_alpha_sigma(ts)

        x = jax.random.normal(rng.split(), [batch_size, 3, image_size[1], image_size[0]])

        if init_array is not None:
            x = sigmas[0] * x + alphas[0] * init_array

        # Main loop
        local_steps = schedule.shape[0] - 1
        for j in tqdm(range(local_steps)):
            if use_panorama:
                # == Panorama ==
                shift = jax.random.randint(rng.split(), [batch_size, 2], 0, jnp.array([1, image_size[0]]))
                x = xyroll(x, shift) 
                # == -------- ==
            if ts[j] == ts[j+1]:
              continue
            # Skip steps where the ts are the same, to make it easier to
            # make complicated schedules out of cat'ing linspaces.
            # diffusion.set(clip_embed=jax.random.normal(rng.split(), [batch_size,512]))
            x, pred = sample_step(rng.split(), x, ts[j], ts[j+1], diffusion, cond_fn, eta)
            assert x.isfinite().all().item()
            if j % displayRate == 0 or j == local_steps - 1:
                images = pred
                # images = jnp.concatenate([images, x], axis=0)
                images = images.add(1).div(2).clamp(0, 1)
                images = torch.tensor(np.array(images))
                if save_intermediate_frames and not j / local_steps > 0.99 and not j == 0:
                    for k in range(batch_size):
                        pil_image = TF.to_pil_image(images[k])
                        intermediateTitle = sanitize(title[k])
                        os.makedirs(f'{outputFolder}/{hmTimeStringFolder}-{intermediateTitle}', exist_ok=True)
                        pil_image.save(f'{outputFolder}/{hmTimeStringFolder}-{intermediateTitle}/step_{j}-{k}-{intermediateTitle}.jpg', quality=90)
            if j % saveEvery == 0 and saveVideo:
                if batch_size < 4:
                    for k in range(batch_size):
                        images = pred.add(1).div(2).clamp(0, 1)
                        images = torch.tensor(np.array(images))
                        stepnum = f'{frameIteration}.png'
                        pil_image = TF.to_pil_image(images[k])
                        pil_image.save(Loc(f'imagesteps/{i}/{k}/'+stepnum))
                else:
                    images = pred
                    images = images.add(1).div(2).clamp(0, 1)
                    images = torch.tensor(np.array(images))
                    TF.to_pil_image(utils.make_grid(images, nColumns).cpu()).save(Loc(f'/imagesteps/{i}/0/{frameIteration}.png'))
                frameIteration += 1
        
        os.makedirs(outputFolder, exist_ok=True)
        for k in range(batch_size):
            this_title = sanitize(title[k])
            pil_image = TF.to_pil_image(images[k])
            index = i * batch_size + k + 1
            index_size = n_batches * batch_size
            pil_image.save(f'{outputFolder}{this_title} ({index} of {index_size}) at {timestring}.jpg', quality=90)
            if saveVideo and batch_size < 4:
                make_video(i, k, saveEvery, secondsOfVideo, batch_size)
        
    print(f'Finished {splitprompts}' if splitprompt > 0 else f'Finished run of ({title[0]}) with seed {local_seed}...')


try:
  run()
  success = True
except:
  import traceback
  traceback.print_exc()
  success = False
assert success
