//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI
import MLXLMCommon

@Observable
@MainActor
class ModelManager {
    var selectedModelType: ModelType = .fastVLM
    var currentModel: any VLMModelProtocol
    
    // Model switching state
    var isSwitchingModels = false
    var switchingProgress: String = ""
    
    init() {
        // Initialize currentModel directly with the default model type
        self.currentModel = ModelFactory.createModel(type: .fastVLM)
    }
    
    func switchModel(to modelType: ModelType) async {
        guard modelType != selectedModelType else { return }
        
        isSwitchingModels = true
        switchingProgress = "Switching to \(modelType.displayName)..."
        
        // Cancel any current generation
        currentModel.cancel()
        
        // Create new model
        selectedModelType = modelType
        currentModel = ModelFactory.createModel(type: modelType)
        
        switchingProgress = "Loading \(modelType.displayName)..."
        
        // Load the new model
        await currentModel.load()
        
        isSwitchingModels = false
        switchingProgress = ""
    }
    
    func loadCurrentModel() async {
        await currentModel.load()
    }
    
    func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        return await currentModel.generate(userInput)
    }
    
    func cancel() {
        currentModel.cancel()
    }
    
    // Computed properties for easy access
    var running: Bool {
        currentModel.running
    }
    
    var modelInfo: String {
        currentModel.modelInfo
    }
    
    var output: String {
        currentModel.output
    }
    
    var promptTime: String {
        currentModel.promptTime
    }
}
