// ComfyBoxGallery — the gallery service process.
//
// Deliberately separate from the ComfyBox engine binary: the engine is a GPU
// process that loads models, gets rebuilt and re-signed, and whose restart
// orphans in-flight jobs. The gallery must stay up across all of that.

import ComfyBoxCatalog

GalleryServer.runCLIEntryPoint(args: Array(CommandLine.arguments.dropFirst()))
