/**
 * Builds the published site: the project page at /skypiano/, with the explorer
 * nested under it at /skypiano/explorer/.
 *
 * The two apps stay separate builds on purpose — the project page is
 * Tailwind/shadcn and the explorer is MUI/three.js, and keeping them apart
 * avoids reconciling two styling systems in one bundle. This script is what
 * makes them one site.
 */
import { execSync } from 'node:child_process';
import { cp, rm, access } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const projectPage = resolve(here, '..');
const explorer = resolve(projectPage, '..', 'explorer');

// A fixed command string, so the shell npm needs on Windows carries nothing
// user-supplied.
const run = (cwd) => execSync('npm run build', { cwd, stdio: 'inherit' });

await access(explorer).catch(() => {
  throw new Error(`Explorer app not found at ${explorer}`);
});

console.log('\n▸ Building the project page…');
run(projectPage);

console.log('\n▸ Building the explorer…');
run(explorer);

const target = resolve(projectPage, 'dist', 'explorer');
await rm(target, { recursive: true, force: true });
await cp(resolve(explorer, 'dist'), target, { recursive: true });

console.log(`\n✓ Site assembled at ${resolve(projectPage, 'dist')} (explorer nested at /explorer/)`);
