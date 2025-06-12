//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
#if canImport(UIKit)
import UIKit
#endif
import MLXLMCommon

@MainActor
protocol VLMModelProtocol: ObservableObject {
    var running: Bool { get set }
    var modelInfo: String { get set }
    var output: String { get set }
    var promptTime: String { get set }
    
    func load() async
    func generate(_ userInput: UserInput) async -> Task<Void, Never>
    func cancel()
}

enum ModelType: String, CaseIterable {
    case fastVLM = "FastVLM"
    case smolVLM = "SmolVLM2"
    
    var displayName: String {
        switch self {
        case .fastVLM:
            return "FastVLM (Core ML + MLX)"
        case .smolVLM:
            return "SmolVLM2 (MLX Native)"
        }
    }
}

@MainActor
class ModelFactory {
    static func createModel(type: ModelType) -> any VLMModelProtocol {
        switch type {
        case .fastVLM:
            return FastVLMModel()
        case .smolVLM:
            return SmolVLMModel()
        }
    }
}
