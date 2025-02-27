{
  description = "A Nix-flake-based Julia development environment";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          venvDir = ".venv";
          packages = with pkgs; [ python311 ] ++
            (with pkgs; [
              julia
            ]);
            shellHook = ''
              # Install packages from Project.toml if it exists
              #!/bin/bash
              julia --project=. -e 'import Pkg; Pkg.add(url="https://github.com/zsundberg/DMUStudent.jl")'
              julia --project=. -e 'import Pkg; Pkg.add("LinearAlgebra")'
              julia --project=. -e 'import Pkg; Pkg.add("POMDPs")'
              julia --project=. -e 'import Pkg; Pkg.add("POMDPTools")'
              julia --project=. -e 'import Pkg; Pkg.add("Plots")'
              printf "\nDone! Enjoy your Julia shell.\n\n"
          '';
        };
      });
    };
}
