#!/usr/bin/env python3
"""Shell custody controls only; fake cargo does not certify the real EOS gate."""
import base64
import itertools
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('cpu-test-positive-control.sh')
ACTUAL = 'lloom/proptest-regressions/token_ring_fuzz.txt'
LEGACY = 'lloom/tests/token_ring_fuzz.proptest-regressions'
FAKE_CARGO = r'''#!/usr/bin/env python3
import os,sys,signal,subprocess,time
from pathlib import Path
state=Path(os.environ['CONTROL_STATE']); n=int(state.read_text())if state.exists()else 0;state.write_text(str(n+1))
scenario=os.environ['CONTROL_SCENARIO']
if n==0 and scenario=='baseline-red':
 print('error: baseline compile failure');sys.exit(23)
if n==1:
 for name in ['lloom/proptest-regressions/token_ring_fuzz.txt','lloom/tests/token_ring_fuzz.proptest-regressions']:
  p=Path(name);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('ab')as f:f.write(b'new-mutant-seed\n')
 if scenario=='term':os.kill(int(Path(os.environ['CONTROL_PID_FILE']).read_text()),signal.SIGTERM);sys.exit(143)
 if scenario=='delayed-term':
  writer="import signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path('writer-ready').write_text('ready');time.sleep(.7);p=Path('lloom/src/generation/token_ring.rs');p.write_bytes(p.read_bytes()+b'late-writer\\n');p=Path('lloom/proptest-regressions/token_ring_fuzz.txt');p.write_bytes(p.read_bytes()+b'late-seed\\n')"
  subprocess.Popen([sys.executable,'-c',writer])
  while not Path('writer-ready').exists():time.sleep(.005)
  os.kill(int(Path(os.environ['CONTROL_PID_FILE']).read_text()),signal.SIGTERM)
  time.sleep(.1);sys.exit(143)
 if scenario=='compile-red':print('error[E0000]: deliberate compile control');sys.exit(101)
 if scenario=='mutant-green':sys.exit(0)
 print('        FAIL [   0.001s] ( 1/17) lloom::main token_ring_fuzz::ring_matches_its_model')
 print("thread 'token_ring_fuzz::ring_matches_its_model' panicked: assertion `left == right` failed")
 print('     Summary [   0.002s] 17 tests run: 12 passed, 5 failed')
 sys.exit(1)
if n==2 and scenario=='restore-red':
 print('error: restored compile control');sys.exit(101)
print('        PASS [   0.001s] ( 1/17) lloom::main token_ring_fuzz::ring_matches_its_model')
print('        PASS [   0.001s] ( 2/17) lloom::main token_ring_tests::eos_flag_is_set_device_side_and_clears')
print('     Summary [   0.002s] 17 tests run: 17 passed')
'''


class Custody(unittest.TestCase):
    def case(self, scenario, exists, old=False):
        # Keep this small temporary reproduction and all shell receipts when a
        # control fails; these are not source checkout/worktree modifications.
        root = Path(tempfile.mkdtemp(prefix='lloom-cpu-custody-'))
        candle = root / 'candle'
        script = candle / '.github/ci/cpu-test-positive-control.sh'
        script.parent.mkdir(parents=True)
        if old:
            # Sensitivity ablation: remove ONLY child-group supervision from
            # the actual script; do not depend on an old checkout or tmp path.
            direct = SCRIPT.read_text()
            begin = direct.index('  active_receipt="$RECEIPTS/$phase-group"', direct.index('check() {'))
            end = direct.index('  cat "$RECEIPTS/$phase.log"', begin)
            direct = direct[:begin]+'  if "${CHECK[@]}" >"$RECEIPTS/$phase.log" 2>&1; then rc=0; else rc=$?; fi\n'+direct[end:]
            script.write_text(direct)
        else:
            shutil.copyfile(SCRIPT, script)
            shutil.copyfile(SCRIPT.with_name('cpu-command-group.py'), script.with_name('cpu-command-group.py'))
        target = candle / 'lloom/src/generation/token_ring.rs'
        target.parent.mkdir(parents=True)
        original = b'before .eq(&self.eos_ids)? after\n'
        target.write_bytes(original)
        for name, present in zip([ACTUAL, LEGACY], exists):
            path = candle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if present:
                path.write_bytes(b'original-seed\x00\n')
        before = {name: (candle / name).read_bytes() if (candle / name).exists() else None for name in [ACTUAL, LEGACY]}
        bin_dir = root / 'bin'
        bin_dir.mkdir()
        cargo = bin_dir / 'cargo'
        cargo.write_text(FAKE_CARGO)
        cargo.chmod(0o755)
        tmp = root / 'receipts'
        tmp.mkdir()
        env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'], TMPDIR=str(tmp), CONTROL_SCENARIO=scenario, CONTROL_STATE=str(root/'calls'), CONTROL_PID_FILE=str(root/'shell-pid'))
        child = subprocess.Popen(['/bin/bash', str(script)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (root/'shell-pid').write_text(str(child.pid))
        stdout,stderr=child.communicate(timeout=45)
        proc = subprocess.CompletedProcess(child.args, child.returncode, stdout, stderr)
        if scenario=='delayed-term':
            import time
            time.sleep(1)
        if old:
            self.assertNotEqual(target.read_bytes(), original, 'old arrangement failed to expose late writer '+str(root))
            return root
        (root / 'stdout.txt').write_text(proc.stdout)
        (root / 'stderr.txt').write_text(proc.stderr)
        self.assertEqual(proc.returncode, {'normal': 0, 'term': 143, 'delayed-term':143}.get(scenario, 1), str(root)+proc.stdout+proc.stderr)
        self.assertEqual(target.read_bytes(), original, str(root))
        for name, content in before.items():
            path = candle / name
            self.assertEqual(path.read_bytes() if path.exists() else None, content, str(root)+name)
        self.assertIn('both_paths_verified=true', proc.stdout)
        if scenario != 'baseline-red':
            retained = list(tmp.glob('lloom-cpu-control.*/seed-*.mutant'))+list(tmp.glob('lloom-cpu-control.*/seed-*.exit'))
            for i in range(2):
                self.assertTrue(any(path.name.startswith(f'seed-{i}.')and b'new-mutant-seed\n'in path.read_bytes()for path in retained), str(root))
            encoded = re.findall(r'base64_begin\n(.*?)\nPERSISTENCE_RECEIPT base64_end', proc.stdout, re.S)
            decoded = [base64.b64decode(''.join(text.split()), validate=True) for text in encoded]
            for path in retained:
                if b'new-mutant-seed\n' in path.read_bytes():
                    self.assertIn(path.read_bytes(), decoded, str(root))
        if scenario == 'compile-red':
            self.assertIn('mutant did not produce a named runtime assertion failure', proc.stdout)
        return root

    def test_clean_baseline_and_normal_restoration_all_original_states(self):
        for exists in itertools.product([False, True], repeat=2):
            with self.subTest(exists=exists): self.case('normal', exists)

    def test_failures_restore_both_paths_and_reject_compile_only_red(self):
        for scenario in ['baseline-red', 'compile-red', 'mutant-green', 'restore-red']:
            for exists in [(False, False), (True, True)]:
                with self.subTest(scenario=scenario, exists=exists): self.case(scenario, exists)

    def test_delayed_orphan_after_term_cannot_race_restoration(self):
        for exists in [(False,False),(True,True)]:
            with self.subTest(exists=exists):self.case('delayed-term',exists)

    def test_old_arrangement_is_sensitive_to_delayed_orphan_writer(self):
        self.case('delayed-term',(False,False),old=True)

    def test_term_is_terminal_nonzero_and_restores_both_paths(self):
        for exists in [(False, False), (True, True)]:
            with self.subTest(exists=exists): self.case('term', exists)


if __name__ == '__main__':
    unittest.main()
