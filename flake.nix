{
  description = "Flakes-based nix shell for motilitAI based on micromamba";

  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          name = "ComParE22";

          pkgs = import nixpkgs {
            config = {
              # CUDA and other "friends" contain unfree licenses. To install them, you need this line:
              allowUnfree = true;
            };
            inherit system;
          };
        in
        rec {
          defaultPackage = pkgs.buildFHSUserEnv {
            inherit name;

            targetPkgs = pkgs: with pkgs; [
              micromamba
              libGL
              zlib
              starship
            ];

            profile = ''
              eval "$(micromamba shell hook -s bash)"
              micromamba create -q -n ${name} python=3.7
              micromamba activate ${name}              
            '';
            runScript = ''
              bash --rcfile <(echo 'eval "$(micromamba shell hook -s bash)"; eval "$(starship init bash)"')
            '';
          };
        }
      );
}

