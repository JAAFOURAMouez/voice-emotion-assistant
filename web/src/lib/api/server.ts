/**
 * Where the voxemo server is. Served by the server itself (the usual case), the page
 * talks to its own origin. Hosted elsewhere (e.g. a static site on Render), it talks
 * to the user's own computer through a tunnel: address and access code are kept in
 * this browser only.
 */

const ADDRESS_KEY = "voxemo.server";
const CODE_KEY = "voxemo.code";

/** Built for a static host: there is no API at the page's own origin. */
export const standalone = import.meta.env.VITE_STANDALONE === "true";
const builtAddress: string = import.meta.env.VITE_API_URL ?? "";

function read(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function write(key: string, value: string): void {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* private window: kept for this page only */
  }
}

let address = read(ADDRESS_KEY) || builtAddress;
let code = read(CODE_KEY);

/** "https://pc.example.ts.net" from what the user typed; "" = this page's origin. */
export function normalizeAddress(raw: string): string {
  const text = raw.trim().replace(/\/+$/, "");
  if (!text) return "";
  const withScheme = /^https?:\/\//i.test(text) ? text : `https://${text}`;
  const url = new URL(withScheme); // throws on nonsense
  const host = url.hostname;
  if (host !== "localhost" && !host.includes(".") && !host.includes(":")) throw new Error("bad address");
  return url.origin;
}

export function serverAddress(): string {
  return address;
}

export function accessCode(): string {
  return code;
}

export function configure(newAddress: string, newCode: string): void {
  address = normalizeAddress(newAddress);
  code = newCode.trim();
  write(ADDRESS_KEY, address === builtAddress ? "" : address);
  write(CODE_KEY, code);
}

/** Nothing to talk to yet: the settings panel must ask for the address. */
export function needsAddress(): boolean {
  return standalone && !address;
}

export function apiUrl(path: string): string {
  return `${address}${path}`;
}

export function socketUrl(path = "/api/turn"): string {
  const base = address || `${location.protocol}//${location.host}`;
  return base.replace(/^http/, "ws") + path;
}

/** WebSocket subprotocols: the access code travels here (no headers on a WebSocket). */
export function socketProtocols(): string[] {
  return code ? ["voxemo", `token.${code}`] : ["voxemo"];
}

export class AccessDenied extends Error {}

export class ApiError extends Error {}

/** fetch() towards the server, with the access code; errors carry the French detail. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (code) headers.set("Authorization", `Bearer ${code}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(apiUrl(path), { ...init, headers });
  if (response.status === 401) throw new AccessDenied();
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail);
  }
  return (await response.json()) as T;
}
