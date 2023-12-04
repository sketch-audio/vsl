// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "vsl",
    products: [
        .library(name: "vsl", targets: ["vsl"])
    ],
    targets: [
        .target(name: "vsl", dependencies: [])
    ],
    cxxLanguageStandard: .cxx20
)
