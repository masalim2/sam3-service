import io
import tarfile
from io import BytesIO


def add_member(tf: tarfile.TarFile, filename: str, buf: BytesIO) -> None:
    info = tarfile.TarInfo(filename)
    buf.seek(0, io.SEEK_END)
    info.size = buf.tell()

    buf.seek(0)
    tf.addfile(info, buf)
