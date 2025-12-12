# Changelog

## Latest
- Refactored ME-IIS into modular components (latent probability backend, joint constraint builder, IIS updater, entropy filter) while keeping Pal & Miller Eq. (14)-(18) intact.
- Added configurable spherical `vmf_softmax` latent backend alongside the existing GMM backend; constraint math still uses class-conditioned joints.
- IIS update denominator now explicitly uses `N_d + N_c` (with `N_d=0`, `N_c` = number of latent variables) as in Eq. (18).
- Centralized CLI parsing into shared builders/configs with auto-generated reference (`docs/cli_reference.md`) and updated README menu.
