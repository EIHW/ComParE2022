# **DeepLearningGPU**
This is a fully fledged-out nix shell which makes use of [nix-flakes](https://nixos.wiki/wiki/Flakes) for automatically pinning and locking dependencies, and [mach-nix](https://github.com/DavHau/mach-nix) to handle installing python wheels and sdists in a reproducible and nixified way.

## Usage
Running either `nix develop` (flakes) or `nix-shell` (legacy method through compatibility utils) and passing the path to the directory (where you copied the folder) as argument, will drop you into a shell where common python-based Deep Learning dependencies are available, e.g. CUDA enabled PyTorch, TensorFlow and Transformers. E.g., from this folder:
```bash
nix develop ./
# or
nix-shell ./
```
When you open the shell for the first time, nix will fetch (and install) all dependencies for you, so this might take a while. Afterwards, calls to the shell will be fast (especially if you use `flakes`).

For convenience, this shell is also added to the cluster's local flake registry and you can deploy it with this shorter command:
```bash
nix develop DeepLearningGPU
```

## Adapting the example to your needs
If you want to adapt the shell with additional python dependencies, just put them into `requirements.txt`, system dependencies go in `system-dependencies.nix`. Sometimes, pip wheels will rely on external system libraries and fail to load. You can put the nixpkgs containing these libraries in `missingLibs` (also in `system-dependencies.nix`). `python.nix` contains the `mach-nix.mkPython` call that generates the python environment from your dependencies. You can override where your packages are installed from in this file (wheels, sdists, conda or nixpkgs)

## Scripts and SLURM Jobs
To activate your shell inside your scripts, you have two choices:
- preface the commands you want to execute with `nix develop ./ --command `
- add a shebang to your scripts, e.g. as in `./scripts.sh`

A sample slurm script can be found in `gpujob.sh`.

