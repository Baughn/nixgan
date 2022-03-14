{ pkgs }:

# This attempts to emulate a Colab environment.

let
  pyPkgs = p: with p; [
    pillow
    braceexpand
    requests
    numpy
    jax
    jaxlibWithCuda
    einops
    cbor2
    tqdm
    pytorch
    torchvision
    dm-haiku
    ftfy
    regex
    imageio
  ];
in

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    (python3.withPackages pyPkgs)
    cudatoolkit_11
  ];
}
