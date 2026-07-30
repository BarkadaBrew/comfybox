import Foundation

public enum GalleryServer {
    /// Replaced with the real server in Task 7.
    public static func runCLIEntryPoint(args: [String]) {
        FileHandle.standardError.write(Data("gallery server not yet implemented\n".utf8))
        exit(1)
    }
}
