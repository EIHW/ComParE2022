{ pkgs, ... }:
{
    additionalPackages = with pkgs; [
    ];
    missingLibs = with pkgs; [
        pkgs.stdenv.cc.cc
    ];
}