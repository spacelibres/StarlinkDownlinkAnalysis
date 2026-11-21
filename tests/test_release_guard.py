"""Release policy tests, including real Git hooks in an isolated temporary repo."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SECTIONS=['Purpose','Changes','Scientific record','Validation','Limitations']

def entry(version='0.0.3'):
    return f'## [{version}] - 2026-11-20\n\n'+''.join(
        f'### {name}\n- Explicit recorded evidence and checks for this milestone.\n\n' for name in SECTIONS)

def load_guard():
    spec=importlib.util.spec_from_file_location('release_guard',ROOT/'tools/release_guard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class ReleasePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.guard=load_guard()

    def test_exact_patch_and_complete_record_pass(self):
        self.guard.validate('0.0.2','0.0.3','# Changelog\n\n'+entry(),subject='v0.0.3: enforce scientific records')

    def test_missing_jump_reused_or_wrong_series_fail(self):
        for version in ['0.0.2','0.0.4','0.1.0','1.0.0','0.0.03','0.0.-1','']:
            with self.subTest(version=version),self.assertRaises(ValueError):
                self.guard.validate('0.0.2',version,entry(version))

    def test_matching_latest_changelog_required(self):
        with self.assertRaises(ValueError):self.guard.validate('0.0.2','0.0.3',entry('0.0.2'))

    def test_empty_missing_placeholder_or_duplicate_sections_fail(self):
        text=entry()
        bad=[text.replace('### Validation','### Other'),
             text.replace('### Validation\n- Explicit recorded evidence and checks for this milestone.','### Validation'),
             text.replace('### Validation\n- Explicit recorded evidence and checks for this milestone.','### Validation\n- TBD'),
             text+'### Validation\n- Duplicate checks must be rejected.\n']
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):self.guard.validate('0.0.2','0.0.3',value)

    def test_subject_must_name_exact_version_and_change(self):
        for subject in ['fix things','v0.0.2: change','v0.0.3: ','v0.0.3:']:
            with self.subTest(subject=subject),self.assertRaises(ValueError):
                self.guard.validate('0.0.2','0.0.3',entry(),subject=subject)

    def test_changelog_dates_and_immutable_history(self):
        with self.assertRaises(ValueError):
            self.guard.validate('0.0.2','0.0.3',entry().replace('2026-11-20','2026-02-30'))
        old='# Changelog\n\n'+entry('0.0.2')
        self.guard.validate('0.0.2','0.0.3',entry()+entry('0.0.2'),previous_changelog=old)
        with self.assertRaises(ValueError):
            self.guard.validate('0.0.2','0.0.3',entry()+entry('0.0.2').replace('Explicit','Rewritten'),previous_changelog=old)

class GitHookIntegrationTests(unittest.TestCase):
    def test_prepare_advances_once_and_refuses_double_preparation(self):
        with tempfile.TemporaryDirectory(prefix='starlink-prepare-test-') as temp:
            directory=Path(temp).resolve()
            self.assertNotEqual(directory,ROOT)
            for args in [('init','-q','-b','main'),('config','user.name','Release policy test'),
                         ('config','user.email','test@example.invalid')]:
                subprocess.run(['git',*args],cwd=directory,check=True,capture_output=True)
            (directory/'VERSION').write_text('0.0.2\n',encoding='utf8')
            old='# Changelog\n\n'+entry('0.0.2')
            (directory/'CHANGELOG.md').write_text(old,encoding='utf8')
            subprocess.run(['git','add','VERSION','CHANGELOG.md'],cwd=directory,check=True,capture_output=True)
            subprocess.run(['git','commit','-q','-m','test baseline'],cwd=directory,check=True,capture_output=True)
            (directory/'tools').mkdir()
            for name in ['release.py','release_guard.py']:shutil.copy2(ROOT/'tools'/name,directory/'tools'/name)
            command=[sys.executable,str(directory/'tools/release.py'),'--title','record verified change','--date','2026-11-20']
            for name in ['purpose','changes','scientific-record','validation','limitations']:
                command.extend(['--'+name,'Specific verified evidence for the prepared test release.'])
            first=subprocess.run(command,cwd=directory,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual((directory/'VERSION').read_text().strip(),'0.0.3')
            updated=(directory/'CHANGELOG.md').read_text(encoding='utf8')
            self.assertTrue(updated.endswith(entry('0.0.2')))
            second=subprocess.run(command,cwd=directory,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((directory/'CHANGELOG.md').read_text(encoding='utf8'),updated)

    def test_real_hooks_reject_bad_index_and_accept_valid_commit(self):
        # TemporaryDirectory owns only this verified, newly created test directory.
        with tempfile.TemporaryDirectory(prefix='starlink-release-test-') as temp:
            directory=Path(temp).resolve()
            self.assertNotEqual(directory,ROOT)
            def git(*args,ok=True):
                result=subprocess.run(['git',*args],cwd=directory,capture_output=True,text=True,encoding='utf8')
                if ok:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return result
            git('init','-q','-b','main');git('config','user.name','Release policy test')
            git('config','user.email','test@example.invalid');git('config','core.autocrlf','false')
            (directory/'VERSION').write_text('0.0.2\n',encoding='utf8')
            (directory/'CHANGELOG.md').write_text('# Changelog\n\n'+entry('0.0.2'),encoding='utf8')
            git('add','VERSION','CHANGELOG.md');git('commit','-q','-m','test baseline')
            (directory/'tools').mkdir();(directory/'.githooks').mkdir()
            shutil.copy2(ROOT/'tools/release_guard.py',directory/'tools/release_guard.py')
            for name in ['pre-commit','commit-msg']:
                shutil.copy2(ROOT/'.githooks'/name,directory/'.githooks'/name)
            git('config','core.hooksPath','.githooks')
            (directory/'sample.txt').write_text('A real staged change.\n')
            git('add','sample.txt')
            # Merely fixing the worktree must not fool a guard that reads the index.
            (directory/'VERSION').write_text('0.0.3\n',encoding='utf8')
            (directory/'CHANGELOG.md').write_text('# Changelog\n\n'+entry()+entry('0.0.2'),encoding='utf8')
            result=git('commit','-q','-m','v0.0.3: bad unstaged record',ok=False)
            self.assertNotEqual(result.returncode,0);self.assertIn('VERSION',result.stderr)
            git('add','VERSION','CHANGELOG.md')
            result=git('commit','-q','-m','v0.0.2: wrong version',ok=False)
            self.assertNotEqual(result.returncode,0);self.assertIn('subject',result.stderr)
            git('commit','-q','-m','v0.0.3: record verified test change')
            self.assertEqual(git('show','HEAD:VERSION').stdout.strip(),'0.0.3')
            # Reusing the version after a real commit is rejected as well.
            (directory/'sample.txt').write_text('A second staged change.\n')
            git('add','sample.txt')
            self.assertNotEqual(git('commit','-q','-m','v0.0.3: reuse forbidden',ok=False).returncode,0)

if __name__=='__main__':unittest.main()
