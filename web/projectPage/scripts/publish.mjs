/**
 * Publishes the assembled site to the gh-pages branch.
 *
 * This builds the branch from scratch in a temporary directory and force-pushes
 * it, rather than using the gh-pages package. That package reuses a cached
 * clone, and when the cache holds a checkout of the default branch it commits
 * the repository's own files (pipeline/, web/, dotfiles) alongside the site.
 * A branch that only ever contains build output cannot drift that way, and
 * force-pushing a generated branch loses nothing.
 */
import { execFileSync, execSync } from 'node:child_process';
import { cp, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const projectPage = resolve(here, '..');
const dist = join(projectPage, 'dist');
const BRANCH = 'gh-pages';

const git = (args, cwd) =>
  execFileSync('git', args, { cwd, encoding: 'utf8' }).trim();

console.log('▸ Building the site…');
execSync('npm run build:site', { cwd: projectPage, stdio: 'inherit' });

const remote = git(['remote', 'get-url', 'origin'], projectPage);
const staging = await mkdtemp(join(tmpdir(), 'skypiano-gh-pages-'));

try {
  await cp(dist, staging, { recursive: true });

  git(['init', '-q', '-b', BRANCH], staging);
  git(['add', '-A'], staging);
  git(
    [
      '-c', 'user.name=SKY-Piano site',
      '-c', 'user.email=noreply@users.noreply.github.com',
      'commit', '-q', '-m', 'Publish the project page with the nested explorer',
    ],
    staging,
  );

  const files = git(['ls-files'], staging).split('\n').filter(Boolean).length;
  console.log(`\n▸ Publishing ${files} files to ${BRANCH}…`);
  execFileSync('git', ['push', '--force', '--quiet', remote, BRANCH], {
    cwd: staging,
    stdio: 'inherit',
  });

  console.log('\n✓ Published. https://joonhyungbae.github.io/skypiano/');
} finally {
  await rm(staging, { recursive: true, force: true });
}
