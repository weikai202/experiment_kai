import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runtime_launcher', str(Path(__file__).with_name('secret_launcher.py')))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

class CredentialsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        launcher.DIRECTORY = Path(self.tmp.name)/'private'
        launcher.CREDENTIAL = launcher.DIRECTORY/'credentials.json'
    def tearDown(self):
        self.tmp.cleanup()
    def test_merge_preserves_both_credentials_and_permissions(self):
        launcher.save_credential('OPENAI_API_KEY','synthetic-openai')
        launcher.save_credential('RAPID_API_KEY','synthetic-rapid')
        launcher.save_credential('OPENAI_API_KEY','synthetic-openai-updated')
        self.assertEqual(launcher.load_saved_credentials(), {'OPENAI_API_KEY':'synthetic-openai-updated','RAPID_API_KEY':'synthetic-rapid'})
        self.assertEqual(launcher.DIRECTORY.stat().st_mode & 0o777,0o700)
        self.assertEqual(launcher.CREDENTIAL.stat().st_mode & 0o777,0o600)
    def test_unsafe_file_mode_rejected(self):
        launcher.save_credential('OPENAI_API_KEY','synthetic-openai')
        launcher.CREDENTIAL.chmod(0o644)
        with self.assertRaises(RuntimeError): launcher.save_credential('RAPID_API_KEY','synthetic-rapid')
    def test_symlink_rejected(self):
        launcher.private_directory()
        target=Path(self.tmp.name)/'outside';target.write_text('{}');target.chmod(0o600)
        launcher.CREDENTIAL.symlink_to(target)
        with self.assertRaises(OSError): launcher.load_saved_credentials()
    def test_hidden_mode_does_not_print_value(self):
        with patch.object(launcher.sys,'argv',['launcher','configure-rapidapi']), patch.object(launcher.sys.stdin,'isatty',return_value=True), patch.object(launcher.getpass,'getpass',return_value='synthetic-hidden-rapid'), patch.object(launcher.sys,'stdout',new_callable=io.StringIO) as output:
            launcher.main()
            self.assertNotIn('synthetic-hidden-rapid',output.getvalue())
        self.assertEqual(launcher.load_saved_credentials()['RAPID_API_KEY'],'synthetic-hidden-rapid')
    def test_noninteractive_configuration_rejected(self):
        with patch.object(launcher.sys,'argv',['launcher','configure-rapidapi']), patch.object(launcher.sys.stdin,'isatty',return_value=False), patch.object(launcher.getpass,'getpass') as prompt:
            with self.assertRaises(RuntimeError): launcher.main()
            prompt.assert_not_called()
    def test_external_preflight_injects_only_rapidapi(self):
        launcher.save_credential('RAPID_API_KEY','synthetic-rapid')
        with patch.dict(launcher.os.environ, {'OPENAI_API_KEY':'synthetic-openai'}, clear=True), patch.object(launcher.sys,'argv',['launcher','external-preflight']), patch.object(launcher.os,'chdir'), patch.object(launcher.os,'execvpe') as execute:
            launcher.main()
            environment=execute.call_args.args[2]
            self.assertEqual(environment['RAPID_API_KEY'],'synthetic-rapid')
            self.assertNotIn('OPENAI_API_KEY',environment)
            self.assertEqual(execute.call_args.args[1],launcher.COMMANDS['external-preflight'])

    def test_saved_key_wins_over_stale_service_environment(self):
        launcher.save_credential('OPENAI_API_KEY','synthetic-current')
        with patch.dict(launcher.os.environ, {'OPENAI_API_KEY':'synthetic-stale'}, clear=True), patch.object(launcher.sys,'argv',['launcher','preflight']), patch.object(launcher.os,'chdir'), patch.object(launcher.os,'execvpe') as execute:
            launcher.main()
            self.assertEqual(execute.call_args.args[2]['OPENAI_API_KEY'],'synthetic-current')

    def test_invalid_value_preserves_existing_file(self):
        launcher.save_credential('OPENAI_API_KEY','synthetic-openai');before=launcher.CREDENTIAL.read_bytes()
        with self.assertRaises(RuntimeError):launcher.save_credential('RAPID_API_KEY','invalid whitespace')
        self.assertEqual(before,launcher.CREDENTIAL.read_bytes())

if __name__=='__main__':unittest.main()
