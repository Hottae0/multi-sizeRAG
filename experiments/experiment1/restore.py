"""Restore an Experiment 1 ZIP or tar.gz into a fresh directory without executing it."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import zipfile
from . import core

MAX_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 20000


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or path.is_absolute() or '..' in path.parts
            or any(':' in p for p in path.parts)):
        raise ValueError('Unsafe archive path')
    return path


def restore(archive, out):
    archive, out = Path(archive), Path(out)
    if out.exists():
        raise ValueError('Use a new restore directory; never overwrite existing results')
    if not archive.is_file() or archive.is_symlink() or archive.stat().st_size > MAX_BYTES:
        raise ValueError('Expected a bounded regular archive file')
    data = archive.read_bytes()
    original_hash = hashlib.sha256(data).hexdigest()
    entries = []
    # The supplied ZIP wraps one tar.gz. Also support direct tar.gz or plain ZIP.
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = z.infolist()
            if len(members) > MAX_MEMBERS or sum(i.file_size for i in members) > MAX_BYTES:
                raise ValueError('ZIP expansion limit exceeded')
            for i in members:
                safe_name(i.filename)
                mode = i.external_attr >> 16
                if i.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError('Encrypted/link entries not supported')
                if i.compress_size and i.file_size / i.compress_size > 1000:
                    raise ValueError('ZIP compression ratio limit exceeded')
            files = [i for i in members if not i.is_dir()]
            if len(files) == 1 and files[0].filename.endswith(('.tar.gz', '.tgz')):
                data = z.read(files[0])
            else:
                entries = [(i.filename, z.read(i)) for i in files]
    if not entries:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
            total = 0
            for index, m in enumerate(t):
                safe_name(m.name)
                if index >= MAX_MEMBERS or not (m.isfile() or m.isdir()):
                    raise ValueError('Archive member limit or unsupported entry')
                total += m.size
                if total > MAX_BYTES:
                    raise ValueError('TAR expansion limit exceeded')
                if m.isfile():
                    entries.append((m.name, t.extractfile(m).read()))
    paths = [str(safe_name(name)) for name, _ in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate archive paths')
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.restore-', dir=out.parent))
    try:
        for (name, content), relative in zip(entries, paths):
            p = staging / 'source' / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        runs = []
        for manifest in sorted((staging / 'source').rglob('manifest.json')):
            if not (manifest.parent / 'raw').is_dir():
                continue
            m = core.validate_manifest(core.read(manifest))
            planned = {j['job_id'] for j in m['jobs']}
            recorded = {p.stem for p in (manifest.parent / 'raw').glob('*.json')}
            runs.append(dict(path=str(manifest.parent.relative_to(staging)), planned_calls=len(planned),
                             recorded_planned_calls=len(planned & recorded), missing_calls=len(planned - recorded)))
        if not runs:
            raise ValueError('No valid manifest/raw run found in archive')
        result = dict(archive_name=archive.name, archive_sha256=original_hash,
                      restored_files=len(entries), runs=runs,
                      file_sha256={name: hashlib.sha256(b).hexdigest() for name, b in entries})
        core.save(staging / 'restore_manifest.json', result)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {k: v for k, v in result.items() if k != 'file_sha256'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(restore(a.archive, a.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
