#!/usr/bin/env python3
"""Restore direct-object linking for isolated allocator B/C; never execute CUDA."""

import collections
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


base = Path('/tmp/cucascade-pinned-pr-20261012')
previous = base / 'sirius-build'
directory = base / 'sirius-build-corrected'
original = json.loads((previous / 'build-metadata.json').read_text())
prefix = Path('/home/kkristensen/Code/sirius/.pixi/envs/default')
donor = Path(original['donor_build'])
ar = prefix / 'bin/aarch64-conda-linux-gnu-ar'
metadata = {
    'method': 'reuse frozen B/C objects; restore direct Sirius object placement',
    'reason': '--exclude-libs,ALL hid RMM registry when direct objects became archive members',
    'previous_metadata': str(previous / 'build-metadata.json'),
    'commands': [],
    'arms': {},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def capture(argv):
    return subprocess.check_output([str(x) for x in argv], text=True)


def run(argv, cwd=directory):
    argv = [str(x) for x in argv]
    metadata['commands'].append({'cwd': str(cwd), 'argv': argv})
    print('RUN', shlex.join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def archive_members(path):
    """Read GNU ar without collapsing duplicate member names."""
    raw = Path(path).read_bytes()
    assert raw[:8] == b'!<arch>\n', path
    offset, long_names, members = 8, b'', []
    while offset < len(raw):
        header = raw[offset:offset + 60]
        assert len(header) == 60 and header[58:60] == b'`\n', (path, offset)
        size = int(header[48:58])
        name = header[:16].decode('ascii').strip()
        start = offset + 60
        data = raw[start:start + size]
        assert len(data) == size
        offset = start + size + (size % 2)
        if name in ('/', '/SYM64/'):
            continue
        if name == '//':
            long_names = data
            continue
        if name.startswith('/') and name[1:].isdigit():
            name_start = int(name[1:])
            name_end = long_names.index(b'/\n', name_start)
            name = long_names[name_start:name_end].decode('ascii')
        else:
            name = name.removesuffix('/')
        assert name and '/' not in name, name
        members.append((name, data))
    assert offset == len(raw)
    assert [name for name, _ in members] == capture([ar, 't', path]).splitlines()
    return members


def verify_members(actual, reference, changed):
    assert [x[0] for x in actual] == [x[0] for x in reference]
    assert sum(name == changed for name, _ in actual) == 1
    unchanged = hashlib.sha256()
    count = 0
    for ordinal, ((name, after), (_, before)) in enumerate(zip(actual, reference)):
        if name != changed:
            assert after == before, (ordinal, name)
            unchanged.update(ordinal.to_bytes(4, 'little') + name.encode() + b'\0' + after)
            count += 1
    return {'count': count, 'indexed_contents_sha256': unchanged.hexdigest()}


def registry_symbols(path):
    output = capture(['readelf', '--dyn-syms', '--wide', path])
    rows = [line.strip() for line in output.splitlines()
            if 'get_ref_map' in line or 'ref_map_lock' in line]
    assert len(rows) >= 3, (path, rows)
    assert all('UNIQUE DEFAULT' in line for line in rows), (path, rows)
    return rows


protected = [
    previous / arm / 'sirius.duckdb_extension' for arm in ('baseline', 'candidate')
] + [
    Path('/home/kkristensen/Code/cuCascade') / path for path in (
        'src/memory/small_pinned_host_memory_resource.cpp',
        'include/cucascade/memory/small_pinned_host_memory_resource.hpp',
        'build/pr-reevaluation/libcucascade.so',
    )
] + [
    Path('/home/kkristensen/Code/sirius/build/release/extension/sirius/sirius.duckdb_extension'),
    Path('/tmp/sirius-pr1834-streams-20261007-1940/build/extension/sirius/sirius.duckdb_extension'),
]
before = {str(path): file_sha(path) for path in protected}
metadata['protected_before'] = before
metadata['normal_registry_symbols'] = {
    str(path): registry_symbols(path) for path in protected[-2:]
}

donor_core = archive_members(donor / 'libsirius_core.a')
donor_resource = archive_members(donor / 'cucascade/libcucascade.a')
assert len(donor_core) == 334 and len(donor_resource) == 22
metadata['duplicate_core_names'] = {
    name: count for name, count in collections.Counter(n for n, _ in donor_core).items()
    if count > 1
}

for arm in ('baseline', 'candidate'):
    out = directory / arm
    objects = out / 'objects'
    objects.mkdir(parents=True, exist_ok=True)
    source_core = previous / arm / 'libsirius_core.a'
    source_resource = previous / arm / 'libcucascade.a'
    core = archive_members(source_core)
    resource = archive_members(source_resource)
    core_check = verify_members(core, donor_core, 'sirius_context.cpp.o')
    resource_check = verify_members(resource, donor_resource,
                                    'small_pinned_host_memory_resource.cpp.o')
    extracted = []
    member_records = []
    for ordinal, (name, data) in enumerate(core):
        target = objects / f'{ordinal:03d}' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        assert target.read_bytes() == data
        extracted.append(str(target))
        member_records.append({'ordinal': ordinal, 'name': name, 'path': str(target),
                               'sha256': sha(data), 'bytes': len(data)})
    target_resource = out / 'libcucascade.a'
    shutil.copy2(source_resource, target_resource)
    assert file_sha(target_resource) == file_sha(source_resource)
    extension = out / 'sirius.duckdb_extension'
    command_record = next(
        record for record in original['commands']
        if '-shared' in record['argv'] and str(previous / arm / 'sirius.duckdb_extension')
        in record['argv']
    )
    old_argv = command_record['argv']
    argv, index, replaced = [], 0, False
    while index < len(old_argv):
        item = old_argv[index]
        if item == '-Wl,--push-state,--whole-archive' and index + 2 < len(old_argv) \
                and old_argv[index + 1] == str(source_core):
            assert old_argv[index + 2] == '-Wl,--pop-state'
            argv.extend(extracted)
            index += 3
            replaced = True
            continue
        if item == str(previous / arm / 'sirius.duckdb_extension'):
            item = str(extension)
        elif item == str(source_resource):
            item = str(target_resource)
        elif item.startswith('-Wl,--dependency-file='):
            item = '-Wl,--dependency-file=' + str(out / 'link.d')
        argv.append(item)
        index += 1
    assert replaced
    assert '-Wl,--exclude-libs,ALL' in argv
    assert str(source_core) not in argv
    assert all(str(path) in argv for path in extracted)
    run(argv, cwd=command_record['cwd'])
    append_record = next(
        record for record in original['commands']
        if '-DEXTENSION=' + str(previous / arm / 'sirius.duckdb_extension') in record['argv']
    )
    append_argv = [('-DEXTENSION=' + str(extension))
                   if item.startswith('-DEXTENSION=') else item for item in append_record['argv']]
    run(append_argv)
    metadata['arms'][arm] = {
        'core_members': member_records,
        'unchanged_core_members': core_check,
        'unchanged_resource_members': resource_check,
        'resource_archive_sha256': file_sha(target_resource),
        'extension': str(extension), 'extension_sha256': file_sha(extension),
        'extension_size': extension.stat().st_size,
        'registry_symbols': registry_symbols(extension),
        'allocator_header_sha256': original['arms'][arm]['allocator_header_sha256'],
        'allocator_source_sha256': original['arms'][arm]['allocator_source_sha256'],
        'owner_source_sha256': original['arms'][arm]['owner_source_sha256'],
    }
    (directory / 'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

after = {str(path): file_sha(path) for path in protected}
assert after == before
metadata['protected_after'] = after
assert metadata['arms']['baseline']['unchanged_core_members'] == \
       metadata['arms']['candidate']['unchanged_core_members']
assert metadata['arms']['baseline']['unchanged_resource_members'] == \
       metadata['arms']['candidate']['unchanged_resource_members']
(directory / 'build-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('DONE corrected direct-object B/C; no source compilation or CUDA execution.', flush=True)
