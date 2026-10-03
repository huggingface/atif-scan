"""Finding and fetching inputs: local files, Hugging Face buckets and the Harbor Hub.

- `inputs`: resolve a path, `hf://` URL or manifest to loadable `Source`s (the only input I/O)
- `sync`: mirror remote inputs into the private sync folder
- `layout`: `--inspect`, classify a listing without reading traces
- `harbor`: Harbor run files, Hub listings and job downloads

This layer imports only the data layer (parsing and facts), never checks or reports.
"""

from .inputs import (
    DEFAULT_PATTERN,
    HF_PREFIX,
    Entry,
    Listing,
    Source,
    SourceError,
    file_source,
    list_input,
    normalize,
    resolve,
)

__all__ = [
    "DEFAULT_PATTERN",
    "HF_PREFIX",
    "Entry",
    "Listing",
    "Source",
    "SourceError",
    "file_source",
    "list_input",
    "normalize",
    "resolve",
]
