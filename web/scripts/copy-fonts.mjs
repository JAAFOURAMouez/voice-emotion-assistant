// Copies the OFL fonts from node_modules into src/assets/fonts (committed to the
// repository, so the app never fetches a font from a CDN). Run: npm run fonts
import { copyFileSync, mkdirSync } from "node:fs";

const target = new URL("../src/assets/fonts/", import.meta.url);
mkdirSync(target, { recursive: true });
const files = {
  "@fontsource-variable/fraunces/files/fraunces-latin-full-normal.woff2": "fraunces-latin-full-normal.woff2",
  "@fontsource-variable/fraunces/files/fraunces-latin-full-italic.woff2": "fraunces-latin-full-italic.woff2",
  "@fontsource-variable/fraunces/LICENSE": "LICENSE-Fraunces.txt",
  "@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2": "jetbrains-mono-latin-wght-normal.woff2",
  "@fontsource-variable/jetbrains-mono/LICENSE": "LICENSE-JetBrainsMono.txt",
};
for (const [source, name] of Object.entries(files)) {
  copyFileSync(new URL(`../node_modules/${source}`, import.meta.url), new URL(name, target));
  console.log("copied", name);
}
