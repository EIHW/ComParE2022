{ pkgs, ... }:
{
    additionalPackages = with pkgs; [
        eihw-packages.deepspectrum
        eihw-packages.audeep
        eihw-packages.opensmile
    ];
    missingLibs = with pkgs; [
        pkgs.stdenv.cc.cc
    ];
}