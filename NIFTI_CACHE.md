# Shared MRI caches

Training and validation jobs can share `prepared_root` and `subtraction_root`.
The dataset locks each cache entry for both reads and creation. It writes a
unique temporary `.nii.gz` beside the destination, closes the file, and uses an
atomic rename to publish it. Readers cannot consume a partially written cache.
MONAI, custom preprocessing, and subtraction caches use the same mechanism.

Older jobs wrote directly to the final cache filename and read existing files
without acquiring the writer lock. Concurrent jobs could therefore encounter
`EOFError: Compressed file ended before the end-of-stream marker was reached`.
Interrupted writes could also leave truncated files for later runs.

The updated loader rebuilds a corrupt derived cache under its lock, using the
original source image(s). The warning names the affected cache and sources.
Healthy entries are reused. Source read failures and other I/O errors still
stop the job; the error includes the cache and source paths and preserves the
underlying exception. Source images are never modified, and no cohort entries
are silently dropped.

Before relaunching the experiment grid, stop any remaining jobs running the
old code and update the checkout. Existing broken caches rebuild as they are
encountered; deleting the whole cache is unnecessary. A failed rebuild from
an original NIfTI requires repairing or re-converting that source file.

Run the focused tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_nifti_cache.py' -v
```

Tests use real NIfTI files, deliberate gzip truncation, interrupted writes,
and concurrent processes. Full training and the Helios filesystem must be
verified on the cluster.
