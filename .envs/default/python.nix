{ pkgs, mach-nix }:
mach-nix.mkPython rec {

    requirements = builtins.readFile ./requirements.txt;

    providers.torch = "nixpkgs";
    providers.torchvision = "nixpkgs";
    overridesPost = [(final: prev: {
      torch = prev.pytorch-bin;
      torchvision = prev.torchvision-bin;
    })];

}