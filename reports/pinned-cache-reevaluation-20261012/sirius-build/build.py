#!/usr/bin/env python3
"""Isolated allocator-only Sirius A/B relink. Compilation only; never execute CUDA."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

base = Path('/tmp/cucascade-pinned-pr-20261012')
directory = base / 'sirius-build'
repo = Path('/home/kkristensen/Code/cuCascade')
consumer = Path('/home/kkristensen/Code/sirius')
source = Path('/tmp/sirius-merge-origin-main-20261009/worktree')
build = Path('/tmp/sirius-merge-origin-main-20261009/build')
old = Path('/tmp/sirius-pr1834-streams-20261007-1940/build')
old_source = old.parent / 'worktree'
prefix = consumer / '.pixi/envs/default'
ninja = prefix / 'bin/ninja'
ar = prefix / 'bin/aarch64-conda-linux-gnu-ar'
compiler = prefix / 'bin/aarch64-conda-linux-gnu-c++'
metadata = {'method': 'allocator-only overlay; rebuild allocating/deleting owner in both arms',
            'source': str(source), 'donor_build': str(build), 'commands': [], 'arms': {}}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def capture(command, cwd=None):
    return subprocess.check_output([str(x) for x in command], cwd=cwd, text=True)

def run(command, cwd=directory):
    command = [str(x) for x in command]
    metadata['commands'].append({'cwd': str(cwd), 'argv': command})
    print('RUN', shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

def compile_command(target, suffix):
    output = capture([ninja, '-C', build, '-t', 'commands', target])
    command = shlex.split(next(line for line in output.splitlines()
                              if ' -c ' in line and line.endswith(suffix)))
    if command[0] == 'sccache':
        command = command[1:]
    return command

def rewrite_compile(command, overlay, input_file, output_file):
    result = list(command)
    result[1:1] = ['-I', str(overlay), '-DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER']
    for flag, value in [('-c', input_file), ('-o', output_file),
                        ('-MF', str(output_file) + '.d'), ('-MT', output_file)]:
        index = result.index(flag)
        result[index + 1] = str(value)
    return result

owner_cmd = compile_command('CMakeFiles/sirius_objects.dir/src/sirius_context.cpp.o',
                            'src/sirius_context.cpp')
allocator_cmd = compile_command(
    'cucascade/CMakeFiles/cucascade_objects.dir/src/memory/small_pinned_host_memory_resource.cpp.o',
    'src/memory/small_pinned_host_memory_resource.cpp')
for label, tree in [('consumer', consumer), ('consumer_dependency', consumer/'cucascade'),
                    ('donor_source', source), ('donor_dependency', source/'cucascade'),
                    ('duckdb', old_source/'duckdb'), ('candidate', repo)]:
    metadata[label] = {'commit': capture(['git', 'rev-parse', 'HEAD'], cwd=tree).strip(),
                       'status': capture(['git', 'status', '--short'], cwd=tree),
                       'diff_sha256': hashlib.sha256(subprocess.check_output(
                           ['git', 'diff', '--binary', 'HEAD'], cwd=tree)).hexdigest()}
metadata['baseline_commit'] = capture(['git', 'rev-parse', 'origin/main'], cwd=repo).strip()
metadata['donor_hashes'] = {str(path): sha(path) for path in [
    build/'libsirius_core.a', build/'cucascade/libcucascade.a',
    build/'cucascade/libcucascade_cudf.a',
    build/'CMakeFiles/sirius_objects.dir/src/sirius_context.cpp.o',
    source/'src/sirius_context.cpp', source/'src/memory/numa_small_pinned_mr.hpp',
    source/'cucascade/include/cucascade/memory/fixed_size_host_memory_resource.hpp',
    source/'cucascade/src/memory/fixed_size_host_memory_resource.cpp',
]}

# Both copied archives retain every original member except the explicitly rebuilt one.
for arm, reference in [('baseline', 'origin/main'), ('candidate', 'HEAD')]:
    out = directory / arm
    overlay = out / 'include'
    out.mkdir(parents=True, exist_ok=True)
    header_path = Path('include/cucascade/memory/small_pinned_host_memory_resource.hpp')
    cpp_path = Path('src/memory/small_pinned_host_memory_resource.cpp')
    header = overlay / header_path.relative_to('include')
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_bytes(subprocess.check_output(['git', 'show', f'{reference}:{header_path}'], cwd=repo))
    allocator = out / 'small_pinned_host_memory_resource.cpp'
    allocator.write_bytes(subprocess.check_output(['git', 'show', f'{reference}:{cpp_path}'], cwd=repo))
    owner = out / 'sirius_context.cpp'
    owner.write_bytes((source/'src/sirius_context.cpp').read_bytes())
    allocator_object = out / 'small_pinned_host_memory_resource.cpp.o'
    owner_object = out / 'sirius_context.cpp.o'
    run(rewrite_compile(allocator_cmd, overlay, allocator, allocator_object), cwd=build)
    run(rewrite_compile(owner_cmd, overlay, owner, owner_object), cwd=build)
    core_archive = out / 'libsirius_core.a'
    resource_archive = out / 'libcucascade.a'
    shutil.copy2(build/'libsirius_core.a', core_archive)
    shutil.copy2(build/'cucascade/libcucascade.a', resource_archive)
    run([ar, 'rD', core_archive, owner_object])
    run([ar, 'rD', resource_archive, allocator_object])
    for copied, original, changed in [(core_archive, build/'libsirius_core.a', owner_object.name),
                                     (resource_archive, build/'cucascade/libcucascade.a', allocator_object.name)]:
        names = capture([ar, 't', original]).splitlines()
        assert names.count(changed) == 1
        assert capture([ar, 't', copied]).splitlines() == names
        unchanged_digest = hashlib.sha256()
        for member in names:
            if member == changed:
                continue
            before = subprocess.check_output([str(ar), 'p', str(original), member])
            after = subprocess.check_output([str(ar), 'p', str(copied), member])
            assert before == after, member
            unchanged_digest.update(member.encode() + b'\0' + before)
        metadata['arms'].setdefault(arm, {})[copied.name + '_unchanged_members_sha256'] = unchanged_digest.hexdigest()
    metadata['arms'][arm].update({
        'allocator_header_sha256': sha(header), 'allocator_source_sha256': sha(allocator),
        'owner_source_sha256': sha(owner), 'allocator_object_sha256': sha(allocator_object),
        'owner_object_sha256': sha(owner_object), 'core_archive_sha256': sha(core_archive),
        'resource_archive_sha256': sha(resource_archive),
    })
    (directory/'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

# Compile current wrapper glue against the same DuckDB revision as cached core/parquet objects.
wrapper = directory / 'sirius_extension.cpp'
wrapper.write_bytes((consumer/'sirius-duckdb/src/sirius/sirius_extension.cpp').read_bytes())
wrapper_header = directory / 'sirius_extension.hpp'
wrapper_header.write_bytes((consumer/'sirius-duckdb/src/include/sirius_extension.hpp').read_bytes())
wrapper_object = directory / 'sirius_extension.cpp.o'
wrapper_flags = ['-std=c++20', '-O3', '-DNDEBUG', '-fPIC', '-fvisibility=hidden',
                 '-DDUCKDB_BUILD_LOADABLE_EXTENSION', '-DEXT_VERSION_SIRIUS="dev"',
                 '-I', directory, '-I', source/'include', '-isystem', old_source/'duckdb/src/include',
                 '-isystem', prefix/'include', '-c', wrapper, '-o', wrapper_object]
run([compiler, *wrapper_flags])
metadata['wrapper_source_sha256'] = sha(wrapper)
metadata['wrapper_header_sha256'] = sha(wrapper_header)
metadata['wrapper_object_sha256'] = sha(wrapper_object)

# Keep the known-good matching DuckDB link environment. Replace ALL old Sirius objects with the
# frozen current core archive, which contains its own matching CUDA device-link object.
commands = capture([ninja, '-C', old, '-t', 'commands', 'extension/sirius/sirius.duckdb_extension'])
line = next(line for line in commands.splitlines()
            if ' -shared ' in line and ' -o extension/sirius/sirius.duckdb_extension ' in line)
parsed = shlex.split(line)
first = parsed.index(str(compiler))
last = parsed.index('&&', first)
original_link = parsed[first:last]
metadata['original_link_command'] = original_link
for arm in ['baseline', 'candidate']:
    out = directory / arm
    extension = out / 'sirius.duckdb_extension'
    link = []
    injected = False
    for item in original_link:
        if item.endswith('.o'):
            if not injected:
                link += [str(wrapper_object), '-Wl,--push-state,--whole-archive',
                         str(out/'libsirius_core.a'), '-Wl,--pop-state']
                injected = True
            continue
        replacements = {
            'extension/sirius/sirius.duckdb_extension': str(extension),
            'cucascade/libcucascade.a': str(out/'libcucascade.a'),
            'cucascade/libcucascade_cudf.a': str(build/'cucascade/libcucascade_cudf.a'),
            'cucascade/libcucascade_topology_discovery.a': str(build/'cucascade/libcucascade_topology_discovery.a'),
            'simpatico_codegen/libsimpatico.a': str(build/'simpatico_codegen/libsimpatico.a'),
            'extension/sirius/rust/crates/telemetry/bridge/libtelemetry_bridge.a': str(build/'rust/crates/telemetry/bridge/libtelemetry_bridge.a'),
        }
        item = replacements.get(item, item)
        link.append(item)
    assert injected
    run(link, cwd=old)
    run([prefix/'bin/cmake', '-DABI_TYPE=CPP', '-DEXTENSION='+str(extension),
         '-DPLATFORM_FILE='+str(old/'duckdb_platform_out'), '-DVERSION_FIELD=v1.5.6',
         '-DEXTENSION_VERSION=dev', '-DNULL_FILE='+str(old_source/'duckdb/scripts/null.txt'),
         '-P', old_source/'duckdb/scripts/append_metadata.cmake'])
    metadata['arms'][arm]['extension_sha256'] = sha(extension)
    metadata['arms'][arm]['extension_size'] = extension.stat().st_size
    metadata['arms'][arm]['extension'] = str(extension)
    (directory/'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('DONE isolated baseline and candidate extensions; no GPU execution occurred.', flush=True)
