{
  description = "Pure Nix environment for dance-notation-etl via uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
        buildSystemOverlay = pyproject-build-systems.overlays.default;
        hacks = pkgs.callPackage pyproject-nix.build.hacks { };

        customOverlay = final: prev: {
          pyqt6 = hacks.nixpkgsPrebuilt {
            from = pkgs.python312Packages.pyqt6;
            prev = prev.pyqt6.overrideAttrs (old: {
              passthru = old.passthru // {
                dependencies = pkgs.lib.filterAttrs (name: _: ! pkgs.lib.hasSuffix "-qt6" name) old.passthru.dependencies;
              };
            });
          };
        };

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (
          pkgs.lib.composeManyExtensions [
            buildSystemOverlay
            overlay
            customOverlay
          ]
        );

        venv = pythonSet.mkVirtualEnv "dance-notation-env" workspace.deps.all;

        runtimeLibs = with pkgs; [
          fontconfig dbus freetype zlib libxkbcommon libGL glib stdenv.cc.cc.lib
          libX11 libXi libxcb libXrender libXext
          libXfixes libXcursor libXrandr libSM libICE
          xcbutil xcbutilimage xcbutilkeysyms
          xcbutilrenderutil xcbutilwm xcbutilcursor
        ];

        fonts = with pkgs; [
          dejavu_fonts
          liberation_ttf
          noto-fonts
          noto-fonts-color-emoji
        ];

        fontsConf = pkgs.makeFontsConf {
          fontDirectories = fonts;
        };

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ venv pkgs.uv ] ++ runtimeLibs;

          shellHook = ''
            ln -sfn "${venv}" .venv
            export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"
            export MPLBACKEND="QtAgg"

            export QT_QPA_PLATFORM="xcb"
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.lib.makeLibraryPath runtimeLibs}"
            export QT_PLUGIN_PATH="${pkgs.qt6.qtbase}/lib/qt-6/plugins"
            export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt6.qtbase}/lib/qt-6/plugins/platforms"

            # Fontconfig — use a config that knows about our bundled Nix fonts
            export FONTCONFIG_FILE="${fontsConf}"
            unset FONTCONFIG_SYSROOT
            export XDG_CACHE_HOME="$PWD/.cache/nix-fontconfig"

            unset XDG_DATA_DIRS QML2_IMPORT_PATH QML_IMPORT_PATH
            unset NIXPKGS_QT6_QML_IMPORT_PATH QT_IM_MODULE
            unset QT_SCREEN_SCALE_FACTORS QT_STYLE_OVERRIDE QT_QPA_PLATFORMTHEME
            unset KDE_FULL_SESSION KDE_SESSION_VERSION KDE_APPLICATIONS_AS_SCOPE
            unset XDG_CONFIG_DIRS XDG_CURRENT_DESKTOP XDG_SESSION_DESKTOP

            echo "❄️ Nix environment ready (Qt on XCB, fonts loaded)."
          '';
        };
      });
}