"""
Symlink-safe filesystem access for user-staged (untrusted) files.

Everything below a user's staging directory is controlled by that user, who can
swap any component for a symlink at any time, including after we've validated
the path. So instead of checking a path and then re-opening it by name, we walk
from a trusted *anchor* directory one component at a time with ``O_NOFOLLOW``
and do all further I/O relative to the resulting file descriptors.

- Reads only accept regular, singly-linked files (no symlinks, FIFOs, devices,
  or hardlinks to files elsewhere).
- Writes go to a fresh ``O_EXCL`` temp file which is then renamed into place, so
  a pre-planted symlink or hardlink at the destination is replaced, never
  written through.

Ported from dino-server's ``safe_fs``.
"""

import errno
import io
import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _CLOEXEC
# O_NONBLOCK so opening a planted FIFO can't hang the server; we reject
# non-regular files via fstat right after.
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _CLOEXEC


class UnsafePathError(ValueError):
    """A user-staged path is a symlink, special file, or otherwise not allowed."""


def _check_component(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise UnsafePathError(f"invalid path component: {name!r}")


def open_dir(anchor: str | Path, parts: Sequence[str]) -> int:
    """
    Open ``anchor/parts[0]/.../parts[-1]`` without following any symlink at or
    below anchor. ``anchor`` must be a trusted, already-canonical directory
    path. Returns a directory fd the caller must close.
    """
    try:
        fd = os.open(anchor, _DIR_FLAGS)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafePathError(f"{anchor} is not a real directory") from e
        raise
    try:
        for part in parts:
            _check_component(part)
            try:
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as e:
                if e.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise UnsafePathError(f"{part!r} is not a real directory") from e
                raise
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_file(dir_fd: int, name: str) -> int:
    """
    Open a regular, singly-linked file in ``dir_fd`` for reading without
    following symlinks. Returns a (blocking) fd the caller must close.
    """
    _check_component(name)
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise UnsafePathError(f"{name!r} is a symlink") from e
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsafePathError(f"{name!r} is not a regular file")
        if st.st_nlink != 1:
            raise UnsafePathError(f"{name!r} is hardlinked; refusing to read it")
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_file(anchor: str | Path, parts: Sequence[str]) -> bytes:
    """Read the regular file at ``anchor/parts...`` without following symlinks."""
    if not parts:
        raise UnsafePathError("empty path")
    dir_fd = open_dir(anchor, parts[:-1])
    try:
        fd = open_file(dir_fd, parts[-1])
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "rb") as f:
        return f.read()


@contextmanager
def atomic_write(dir_fd: int, name: str) -> Iterator[BinaryIO]:
    """Write ``name`` in ``dir_fd`` via an exclusive temp file + rename."""
    _check_component(name)
    tmp = f".{name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(tmp, _WRITE_FLAGS, 0o644, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        raise


class PreadReader(io.RawIOBase):
    """
    Seekable, read-only view of an already-opened fd using ``pread``.

    Each reader keeps its own offset, so forked DataLoader workers can share
    one inherited fd (opened once, safely) without racing on the file
    position. Never closes the underlying fd.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = os.fstat(self._fd).st_size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, b) -> int:  # type: ignore[override]
        data = os.pread(self._fd, len(b), self._pos)
        n = len(data)
        b[:n] = data
        self._pos += n
        return n
