// Exercise version selection and the real PyPI build without publishing or credentials.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { cp, mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import semanticRelease from 'semantic-release';
import { prepare, verifyConditions } from 'semantic-release-pypi';

const root = fileURLToPath(new URL('../..', import.meta.url));
const config = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const temp = await mkdtemp(path.join(tmpdir(), 'birdepy-release-'));
const source = path.join(temp, 'source');
const remote = path.join(temp, 'remote.git');
const env = {
  PATH: process.env.PATH,
  GIT_CONFIG_GLOBAL: '/dev/null',
  GIT_CONFIG_NOSYSTEM: '1',
  GIT_AUTHOR_NAME: 'Release check',
  GIT_AUTHOR_EMAIL: 'release-check@example.invalid',
  GIT_COMMITTER_NAME: 'Release check',
  GIT_COMMITTER_EMAIL: 'release-check@example.invalid',
};
const git = (...args) => execFileSync('git', args, { cwd: source, env, encoding: 'utf8' });

try {
  // Include only package inputs, excluding the caller's Git state and credentials.
  await cp(path.join(root, 'src'), path.join(source, 'src'), {
    recursive: true,
    filter: name => !name.includes('__pycache__') && !name.includes('.egg-info'),
  });
  for (const name of ['pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in', 'requirements.txt']) {
    await cp(path.join(root, name), path.join(source, name));
  }
  git('init', '--initial-branch=main');
  git('add', '.');
  git('commit', '-m', 'feat: initial release');
  git('tag', 'v1.0.0');
  git('commit', '--allow-empty', '-m', 'fix: numerical corrections\n\nBREAKING CHANGE: corrected numerical contracts');
  git('init', '--bare', '--initial-branch=main', remote);
  git('remote', 'add', 'origin', remote);
  git('push', '--tags', 'origin', 'main');
  const refsBefore = git('ls-remote', 'origin');
  const result = await semanticRelease({
    ...config.release,
    branches: ['main'],
    repositoryUrl: pathToFileURL(remote).href,
    ci: false,
    dryRun: true,
    // Auth is checked only in the real release job. A PR has no publishing secrets.
    verifyConditions: [],
  }, { cwd: source, env });
  assert.equal(result.nextRelease.version, '2.0.0');
  assert.equal(git('ls-remote', 'origin'), refsBefore, 'Dry run must not create release tags');

  const [, pypiOptions] = config.release.plugins.find(
    plugin => Array.isArray(plugin) && plugin[0] === 'semantic-release-pypi',
  );
  const options = { ...pypiOptions, srcDir: source, pypiPublish: false };
  const context = {
    cwd: source,
    logger: console,
    stdout: process.stdout,
    stderr: process.stderr,
    nextRelease: result.nextRelease,
  };
  await verifyConditions(options, context);
  // semantic-release skips prepare in dry runs; explicitly exercise the build.
  await prepare(options, context);
  const dist = path.join(source, 'dist');
  const artifacts = (await readdir(dist)).map(name => path.join(dist, name));
  assert(artifacts.some(name => name.endsWith('2.0.0-py3-none-any.whl')));
  assert(artifacts.some(name => name.endsWith('2.0.0.tar.gz')));
  execFileSync('python3', ['-m', 'twine', 'check', '--strict', ...artifacts], { stdio: 'inherit' });
  execFileSync('python3', ['-c', `
import sys, zipfile
from email.parser import BytesParser
with zipfile.ZipFile(sys.argv[1]) as wheel:
    assert 'birdepy/data_files/iltcme.json' in wheel.namelist()
    metadata = BytesParser().parsebytes(wheel.read('birdepy-2.0.0.dist-info/METADATA'))
    assert metadata['Version'] == '2.0.0'
    assert metadata['Requires-Python'] == '>=3.10'
    assert not any(name.endswith('entry_points.txt') for name in wheel.namelist())
`, artifacts.find(name => name.endsWith('.whl'))], { stdio: 'inherit' });
  console.log('Release dry run, versioned distributions, and metadata checks passed.');
} finally {
  await rm(temp, { recursive: true, force: true });
}
