# Forma Two-Studio UI Update

This update reorganizes the existing navigation into two clear workspaces without removing existing views or changing backend APIs.

- **Data Studio:** Data Profiling, Data Cleaning, Feature Engineering, Data Explorer.
- **ML Studio:** Pipeline Studio, Experiment Lab, Explainability, Model Registry.

A persistent studio switcher sits above grouped navigation. Selecting a studio opens its primary workspace; selecting an item opens that existing feature. This is a UI information-architecture update only; it does not claim to add persistence or new backend capabilities.

Apply by backing up the project and replacing the corresponding `source/frontend/src/App.tsx` and `source/frontend/src/styles.css` files, or use the full source tree in this archive. Run `npm install` and `npm run build` in `source/frontend` to validate locally.
