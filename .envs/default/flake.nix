{
  description = "Nix shell for ComParE22";

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
          python = (import ./python.nix { inherit pkgs; mach-nix=machNix;});
          systemDependencies = import ./system-dependencies.nix { inherit pkgs; };
          additionalApplications = pkgs.buildEnv {
            inherit name;
            paths = systemDependencies.additionalPackages;
          };
                  
        in
        rec {
          devShell = pkgs.mkShell {
            inherit name;
            PYTHONPATH = "${python}/${python.sitePackages}";
            PYTHONUNBUFFERED = 1;
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemDependencies.missingLibs}:$LD_LIBRARY_PATH";
              unset SOURCE_DATE_EPOCH
            '';
            # ${pkgs.cudaPackages.cudatoolkit_11_2}/lib:
            buildInputs = [
              python
              additionalApplications
              pkgs.jre_minimal
            ];
          };
          packages = flake-utils.lib.flattenTree {
            shell2docker = dockerImage;
          };
          dockerImage = 
          let
              env-shim = pkgs.runCommand "env-shim" {} ''
                mkdir -p $out/usr/bin
                ln -s ${pkgs.coreutils}/bin/env $out/usr/bin/env
              '';
          in
          pkgs.dockerTools.buildLayeredImage {
            name = "mauricege/ComParE22";
            tag = "latest";
            
            contents = [
              python
              pkgs.bash
              pkgs.coreutils
              pkgs.which
              additionalApplications
              systemDependencies.missingLibs
              pkgs.jre_minimal
              env-shim
            ];
            config = {
              Cmd = [ "bash" ];
              Env = [
                "LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath systemDependencies.missingLibs}:/usr/lib64/:$LD_LIBRARY_PATH"
                "PYTHONPATH=${python}/${python.sitePackages}"
                "PYTHONBUFFERED=1"
              ];
            };
          };

        }
      );
}
