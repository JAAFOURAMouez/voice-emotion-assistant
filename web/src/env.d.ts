/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Address of the voxemo server when the page is hosted elsewhere (static site). */
  readonly VITE_API_URL?: string;
  /** "true" on a static host: no API at the page's own origin. */
  readonly VITE_STANDALONE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
