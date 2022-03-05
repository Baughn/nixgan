{
  description = "Nixgan flake";

  # Use the 0.2 branch of jax
  inputs.nixpkgs.url = "github:NixOS/nixpkgs?rev=d6e277169532cc2491af7335585a0773f45fae01";

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
    devShell.${system} = import ./shell.nix { inherit pkgs; };
  };
}
