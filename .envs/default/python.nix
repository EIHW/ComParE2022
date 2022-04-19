{ pkgs, mach-nix }:
mach-nix.mkPython rec {

    requirements = builtins.readFile ./requirements.txt;

    providers.librosa = "nixpkgs";
    providers.soundfile = "nixpkgs";
}