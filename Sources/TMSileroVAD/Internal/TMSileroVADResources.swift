import Foundation

enum TMSileroVADResources {
    /// Resolves the `.mlmodelc` URL by trying:
    ///  1. SPM-generated `Bundle.module` (when consumed via SwiftPM)
    ///  2. CocoaPods `TMSileroVADResources.bundle` next to the framework binary
    ///  3. The framework bundle itself (when the .mlmodelc is direct-included)
    static func modelURL(forName name: String) throws -> URL {
        let candidates: [Bundle?] = [
            cocoapodsResourceBundle(),
            spmModuleBundle(),
            Bundle(for: TMSileroVAD.self)
        ]

        for candidate in candidates.compactMap({ $0 }) {
            if let url = candidate.url(forResource: name, withExtension: "mlmodelc") {
                return url
            }
        }
        throw TMSileroVADError.modelNotFound(name: name)
    }

    private static func cocoapodsResourceBundle() -> Bundle? {
        let candidates: [URL] = [
            Bundle(for: TMSileroVAD.self).resourceURL,
            Bundle.main.resourceURL
        ].compactMap { $0 }
        for url in candidates {
            let bundleURL = url.appendingPathComponent("TMSileroVADResources.bundle")
            if let bundle = Bundle(url: bundleURL) {
                return bundle
            }
        }
        return nil
    }

    private static func spmModuleBundle() -> Bundle? {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return nil
        #endif
    }
}
