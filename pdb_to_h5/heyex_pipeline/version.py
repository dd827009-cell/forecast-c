"""Parser version (semver). Major bump == HDF5 re-parse required."""

parser_version: str = "3.0.0"


def parser_major_version() -> int:
    return int(parser_version.split(".")[0])
