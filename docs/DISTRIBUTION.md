# Source distribution boundary

This public candidate is a new source snapshot. The private research history
and asset corpus remain in a separate local checkout and backup. They must
not be merged or pushed into a repository intended to be public.

The source contains simulator code, bindings, training tools, capture and
local asset-generation tools, synthetic tests and viewer code. Third-party
viewer libraries and fonts retain their own licenses. The original code
license must be selected before publication.

Game packs, executables, raw and converted geometry, vehicle and route
snapshots, image data, captures, replays, model weights and generated scenes
are excluded. Results in the README are reported research results, not a
publicly downloadable verification bundle. Sharing policy artifacts or
captures later needs a separate contents and provenance review.

The physics bump-mask image used to be embedded in a source header. It is now
extracted locally and validated before compilation. Consequently, a library
compiled with local assets is not the same distribution as the source-only
library. Do not publish asset-bearing build directories or CI artifacts.

`tools/audit_public_tree.py` checks the tracked-file inventory for excluded
paths, binary game formats, unexpected binary files and the embedded image's
known byte sequence. CI runs this alongside the asset-free build. It is a
packaging check, not a determination of legal rights in reconstructed code.
Source licensing grants only rights the licensors actually hold.

Before changing repository visibility, replace or remove every branch and tag
that retains the private history, and inspect releases, workflow artifacts,
attachments and other hosted copies. Deleting a file in a new commit is not
history removal. GitHub can retain cached or other references after a force
push; a new repository is preferable when a clean boundary cannot be confirmed.
