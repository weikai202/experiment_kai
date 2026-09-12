"""Fixed-command launcher. Credential values never enter logs or repository files."""
import json
import getpass
import hmac
import os
from pathlib import Path
import stat
import sys
import tempfile

DIRECTORY = Path('/root/.config/toolsandbox-runtime')
CREDENTIAL = DIRECTORY / 'credentials.json'
COMMANDS = {
    'credential-chain-audit': ['uv', 'run', '--frozen', 'python', '/root/toolsandbox-runtime/credential_chain_audit.py'],
    'openai-auth-check': ['uv', 'run', '--frozen', 'python', '/root/toolsandbox-runtime/openai_auth_check.py'],
    'full': ['uv', 'run', '--frozen', 'python', '-m', 'toolsandbox_pipeline.orchestration.live_full_bootstrap'],
    'live-calibration': ['uv', 'run', '--frozen', 'python', '-m', 'toolsandbox_pipeline.orchestration.live_calibration_bootstrap'],
    'external-preflight': ['uv', 'run', '--frozen', 'python', '/root/toolsandbox-runtime/external_preflight.py'],
    'preflight': ['uv', 'run', '--frozen', 'python', '/root/toolsandbox-runtime/unattended_preflight.py'],
    'train-smoke': ['uv', 'run', '--frozen', 'python', '/root/toolsandbox-runtime/train_smoke.py'],
    'reflection-smoke': ['uv', 'run', '--frozen', 'python', '-m', 'toolsandbox_pipeline.orchestration.reflection_smoke'],
}


def private_directory():
    if any(p.is_symlink() for p in (DIRECTORY, *DIRECTORY.parents)):
        raise RuntimeError('credential directory must not be a symlink')
    DIRECTORY.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = DIRECTORY.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError('credential directory requires owner-only permissions')



def load_saved_credentials(*, missing_ok=False):
    private_directory()
    try:
        fd = os.open(CREDENTIAL, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        if missing_ok:
            return {}
        raise
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or not stat.S_ISREG(info.st_mode):
            raise RuntimeError('credential file requires owner-only permissions')
        values = json.load(stream)
    if type(values) is not dict or set(values) - {'OPENAI_API_KEY', 'RAPID_API_KEY'} or any(type(v) is not str or not v for v in values.values()):
        raise RuntimeError('invalid saved credential structure')
    return values


def save_credential(name, value):
    if name not in {'OPENAI_API_KEY', 'RAPID_API_KEY'} or type(value) is not str or not value or any(c.isspace() for c in value):
        raise RuntimeError('invalid credential input')
    values = load_saved_credentials(missing_ok=True)
    values[name] = value
    fd, temporary = tempfile.mkstemp(prefix='.credential-', dir=DIRECTORY)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(values, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CREDENTIAL)
        directory_fd = os.open(DIRECTORY, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    if len(sys.argv) != 2:
        raise RuntimeError('one fixed mode is required')
    mode = sys.argv[1]
    if mode == 'enable-unattended':
        value = os.environ.get('OPENAI_API_KEY', '')
        if not value:
            raise RuntimeError('OPENAI_API_KEY is not configured in this terminal')
        save_credential('OPENAI_API_KEY', value)
        print('Unattended credential loading enabled; file permissions 0600. No key displayed.')
        return
    if mode in {'configure-rapidapi', 'configure-openai'}:
        if not sys.stdin.isatty():
            raise RuntimeError('interactive terminal required')
        name = 'RAPID_API_KEY' if mode == 'configure-rapidapi' else 'OPENAI_API_KEY'
        value = getpass.getpass('Paste ' + name + ' (hidden): ').strip()
        save_credential(name, value)
        print('Credential saved; file permissions 0600. No key displayed.')
        return
    if mode == 'disable-unattended':
        private_directory()
        CREDENTIAL.unlink(missing_ok=True)
        print('Saved credential removed.')
        return
    if mode not in COMMANDS:
        raise RuntimeError('allowed modes: openai-auth-check, full, live-calibration, external-preflight, preflight, train-smoke, reflection-smoke, enable-unattended, configure-rapidapi, configure-openai, disable-unattended')
    env = dict(os.environ)
    # Saved unattended configuration is authoritative over stale service env.
    saved = load_saved_credentials(missing_ok=True)
    if mode == 'credential-chain-audit':
        inherited = env.get('OPENAI_API_KEY', '')
        env['CREDENTIAL_AUDIT_INHERITED_PRESENT'] = '1' if inherited else '0'
        env['CREDENTIAL_AUDIT_INHERITED_MATCH'] = '1' if inherited and hmac.compare_digest(inherited.encode(), saved.get('OPENAI_API_KEY', '').encode()) else '0'
    required = ('RAPID_API_KEY',) if mode == 'external-preflight' else ('OPENAI_API_KEY',)
    if mode in {'live-calibration', 'full'}:
        required = ('OPENAI_API_KEY', 'RAPID_API_KEY')
    for name in required:
        value = saved.get(name) or env.get(name)
        if type(value) is not str or not value:
            raise RuntimeError('required credential is not configured')
        env[name] = value
    if mode == 'external-preflight':
        env.pop('OPENAI_API_KEY', None)
    env.update(TZ='UTC', LC_ALL='C.UTF-8', PYTHONHASHSEED='0')
    os.chdir('/root/toolsandbox')
    os.execvpe(COMMANDS[mode][0], COMMANDS[mode], env)


if __name__ == '__main__':
    os.umask(0o077)
    try:
        main()
    except Exception as error:
        # Fixed categories only; never exception values or local variables.
        print('Launcher failed: ' + type(error).__name__, file=sys.stderr)
        sys.exit(2)
