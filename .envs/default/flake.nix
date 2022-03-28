{
  description = "CUDA enabled Deep Learning Shell";

  # inputs.nixpkgs.url = "nixpkgs/release-21.11";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.flake-compat = {
    url = "github:edolstra/flake-compat";
    flake = false;
  };
  inputs.pypi-deps-db = {
    url = "github:DavHau/pypi-deps-db";
    flake = false;
  };
  inputs.mach-nix = {
    url = "github:DavHau/mach-nix";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.pypi-deps-db.follows = "pypi-deps-db";
    inputs.flake-utils.follows = "flake-utils";

  };
  inputs.eihw-packages = {
    url = "git+https://git.rz.uni-augsburg.de/gerczuma/eihw-packages?ref=main";
    inputs.nixpkgs.follows = "nixpkgs";
  };


  outputs = { self, nixpkgs, mach-nix, flake-utils, pypi-deps-db, eihw-packages, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          name = "ComParE2022";

          pkgs = import nixpkgs {
            config = {
              # CUDA and other "friends" contain unfree licenses. To install them, you need this line:
              allowUnfree = true;
            };
            inherit system;
            overlays = [ eihw-packages.overlay ];
          };
          machNix = import mach-nix {
            python = "python38";
            inherit pkgs;
            pypiData = pypi-deps-db;
          };
          systemDependencies = import ./system-dependencies.nix { inherit pkgs; };
        in
        rec {
          defaultPackage = pkgs.buildEnv {
            inherit name;
            paths = [
              (import ./python.nix { inherit pkgs; mach-nix=machNix;})
            ] ++ systemDependencies.additionalPackages;
          };
          devShell = pkgs.mkShell {
            inherit name;
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemDependencies.missingLibs}:$LD_LIBRARY_PATH";
              unset SOURCE_DATE_EPOCH
            '';
            # ${pkgs.cudaPackages.cudatoolkit_11_2}/lib:
            buildInputs = [
              defaultPackage
            ];
          };
          packages = flake-utils.lib.flattenTree {
            shell2docker = dockerImage;
          };
          dockerImage = pkgs.dockerTools.buildLayeredImage {
            name = "mauriceg/DeepLearningGPU";
            tag = "latest";
            contents = [
              pkgs.bash
              pkgs.coreutils
              defaultPackage
              systemDependencies.missingLibs
            ];
            config = {
              Cmd = [ "dvc" "repro" ];
              Env = [
                "LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath systemDependencies.missingLibs}:/usr/lib64/:$LD_LIBRARY_PATH"
              ];
            };
          };

        }
      );
}
