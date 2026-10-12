#!/usr/bin/env python3
"""Compile/link the prepared observer after root permits CPU load; no GPU execution."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

base = Path('/tmp/cucascade-pinned-pr-20261012')
out = base/'facade-probe'
donor = base/'sirius-build'
metadata = json.loads((donor/'build-metadata.json').read_text())
probe = json.loads((out/'prepare-metadata.json').read_text())
commands = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(command, cwd):
    command = [str(item) for item in command]
    commands.append({'argv': command, 'cwd': str(cwd)})
    print('RUN', shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

# Rebuild only the owner. Its first include directory overlays the instrumented NUMA facade;
# the candidate allocator header and every other frozen input keep their existing paths.
owner = out/'sirius_context.cpp'
object_file = out/'sirius_context.cpp.o'
entry = next(item for item in metadata['commands'] if '-c' in item['argv'] and
             item['argv'][item['argv'].index('-c') + 1].endswith('/candidate/sirius_context.cpp'))
compile_args = list(entry['argv'])
compile_args[1:1] = ['-I', str(out/'include')]
for flag, value in [('-c', owner), ('-o', object_file), ('-MT', object_file),
                    ('-MF', str(object_file)+'.d')]:
    compile_args[compile_args.index(flag) + 1] = str(value)
run(compile_args, entry['cwd'])
archive = out/'libsirius_core.a'
shutil.copy2(donor/'candidate/libsirius_core.a', archive)
ar = Path(compile_args[0]).parent/'aarch64-conda-linux-gnu-ar'
run([ar, 'rD', archive, object_file], out)

# Keep all unchanged archive members, candidate allocator archive, common wrapper glue and
# external dependencies byte-identical to the timing candidate; only the owner is replaced.
original_archive = donor/'candidate/libsirius_core.a'
names = subprocess.check_output([str(ar), 't', str(original_archive)], text=True).splitlines()
assert names.count(object_file.name) == 1
assert subprocess.check_output([str(ar), 't', str(archive)], text=True).splitlines() == names
for member in names:
    if member != object_file.name:
        assert subprocess.check_output([str(ar), 'p', str(archive), member]) == subprocess.check_output([str(ar), 'p', str(original_archive), member]), member
extension = out/'sirius.duckdb_extension'
link_entry = next(item for item in metadata['commands'] if '-shared' in item['argv'] and
                  str(donor/'candidate/sirius.duckdb_extension') in item['argv'])
link_args = [str(archive) if item == str(original_archive) else
             str(extension) if item == str(donor/'candidate/sirius.duckdb_extension') else item
             for item in link_entry['argv']]
run(link_args, link_entry['cwd'])
append_entry = next(item for item in metadata['commands'] if
                    '-DEXTENSION='+str(donor/'candidate/sirius.duckdb_extension') in item['argv'])
append_args = [('-DEXTENSION='+str(extension)) if item ==
               '-DEXTENSION='+str(donor/'candidate/sirius.duckdb_extension') else item
               for item in append_entry['argv']]
run(append_args, append_entry['cwd'])
probe.update({'prepared_only': False, 'compiled_only': True, 'commands': commands,
              'probe_extension': str(extension), 'probe_extension_sha256': sha(extension),
              'probe_owner_object_sha256': sha(object_file),
              'candidate_allocator_archive_sha256': sha(donor/'candidate/libcucascade.a'),
              'unchanged_timing_candidate_extension_sha256': sha(donor/'candidate/sirius.duckdb_extension')})
(out/'build-metadata.json').write_text(json.dumps(probe, indent=2) + '\n')
print('Built observer only; root owns runtime execution, and timing arms remain unchanged.')
