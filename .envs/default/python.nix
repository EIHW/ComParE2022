{ pkgs, mach-nix }:
mach-nix.mkPython rec {

    # you can basically pass in your python requirements as a regular requirements.txt file
    requirements = builtins.readFile ./requirements.txt;

    /*
    As described above, by default, mach-nix uses prebuilt python wheels.
    Unfortunately, some wheels are built such that they expect system libraries in their default FHS paths (which Nix does not use).
    For these cases, you can use the python source distribution "sdist" or even "nixpkgs".
    This has to be done for librosa (and its dependency "soundfile").
    */
    providers.librosa = "nixpkgs";
    providers.soundfile = "nixpkgs";

    providers.gitpython = "nixpkgs";

    providers.torch = "nixpkgs";
    providers.torchvision = "nixpkgs";
    providers.tensorflow = "nixpkgs";
    providers.torchaudio = "nixpkgs";
    providers.tensorboard = "nixpkgs";
    overridesPost = [(final: prev: {
      torch = prev.pytorch-bin;
      torchvision = prev.torchvision-bin;
      torchaudio = prev.torchaudio-bin;
      tensorflow = prev.tensorflow-bin.override { cudaSupport = true; };
      tensorboard = prev.tensorflow-tensorboard;
    })];

}