/**
 * Resolves a site-absolute asset path against the deployment base URL.
 *
 * The app is served from a GitHub Pages project path (`/skypiano/`), so paths
 * written as `/Data/...` or `/samples/...` in the manifest and in the legacy
 * adapters must be prefixed with the base Vite was built with. Paths that are
 * already absolute URLs, or that already carry the base, are left alone.
 */
const BASE = import.meta.env.BASE_URL || '/';

export function withBase(path: string): string {
  if (!path) return path;
  if (/^[a-z]+:\/\//i.test(path)) return path;
  if (!path.startsWith('/')) return path;
  if (BASE === '/') return path;
  const base = BASE.replace(/\/$/, '');
  return path.startsWith(`${base}/`) ? path : `${base}${path}`;
}
