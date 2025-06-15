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
    var selectedModelType: ModelType = .smolVLM
    var currentModel: any VLMModelProtocol
    
    // Model switching state
    var isSwitchingModels = false
    var switchingProgress: String = ""
    
    private var secondaryModel: (any VLMModelProtocol)? = nil
    private var secondaryModelType: ModelType? = nil
    
    init() {
        self.currentModel = ModelFactory.createModel(type: .smolVLM)
        
        // Initialize secondary model (FastVLM) in background
        Task {
            await initializeSecondaryModel()
        }
    }
    
    private func initializeSecondaryModel() async {
        // Create the secondary model (FastVLM when SmolVLM is primary)
        let secondaryType: ModelType = selectedModelType == .smolVLM ? .fastVLM : .smolVLM
        secondaryModel = ModelFactory.createModel(type: secondaryType)
        secondaryModelType = secondaryType
        
        // Load secondary model in background
        await secondaryModel?.load()
    }
    
    func switchModel(to modelType: ModelType) async {
        guard modelType != selectedModelType else { return }
        
        isSwitchingModels = true
        switchingProgress = "Switching to \(modelType.displayName)..."
        
        // Cancel any current generation
        currentModel.cancel()
        
        // If we have the target model already loaded as secondary, swap them
        if let secondaryModel = secondaryModel, secondaryModelType == modelType {
            let oldCurrent = currentModel
            let oldType = selectedModelType
            
            // Swap models
            self.currentModel = secondaryModel
            self.selectedModelType = modelType
            
            // Make old current the new secondary
            self.secondaryModel = oldCurrent
            self.secondaryModelType = oldType
        } else {
            // Create new model if not available
            selectedModelType = modelType
            currentModel = ModelFactory.createModel(type: modelType)
            
            switchingProgress = "Loading \(modelType.displayName)..."
            await currentModel.load()
            
            // Update secondary model
            await initializeSecondaryModel()
        }
        
        isSwitchingModels = false
        switchingProgress = ""
    }
    
    func loadCurrentModel() async {
        await currentModel.load()
    }
    
    func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        if running {
            print("[ModelManager] Cancelling existing generation to start new one")
            currentModel.cancel()
            // Give a small delay to allow cancellation to complete
            try? await Task.sleep(for: .milliseconds(50))
        }
        
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
        let primaryInfo = currentModel.modelInfo
        
        if let secondaryModel = secondaryModel,
           let secondaryType = secondaryModelType {
            let secondaryInfo = secondaryModel.modelInfo
            
            if primaryInfo.isEmpty && secondaryInfo.isEmpty {
                return "Loading models..."
            } else if primaryInfo.isEmpty {
                return "Loading \(selectedModelType.rawValue)... | \(secondaryType.rawValue): \(secondaryInfo)"
            } else if secondaryInfo.isEmpty {
                return "\(selectedModelType.rawValue): \(primaryInfo) | Loading \(secondaryType.rawValue)..."
            } else {
                return "\(selectedModelType.rawValue): \(primaryInfo) | \(secondaryType.rawValue): \(secondaryInfo)"
            }
        }
        
        return primaryInfo.isEmpty ? "Loading \(selectedModelType.rawValue)..." : "\(selectedModelType.rawValue): \(primaryInfo)"
    }
    
    var output: String {
        currentModel.output
    }
    
    var promptTime: String {
        currentModel.promptTime
    }
    
    var secondaryModelInfo: String {
        guard let secondaryModel = secondaryModel,
              let secondaryType = secondaryModelType else {
            return "No secondary model"
        }
        
        let info = secondaryModel.modelInfo
        return info.isEmpty ? "\(secondaryType.rawValue): Loading..." : "\(secondaryType.rawValue): \(info)"
    }
    
    var allModelsStatus: String {
        let primaryStatus = "\(selectedModelType.rawValue): \(running ? "Running" : "Ready")"
        
        if let secondaryType = secondaryModelType {
            let secondaryRunning = secondaryModel?.running ?? false
            let secondaryStatus = "\(secondaryType.rawValue): \(secondaryRunning ? "Running" : "Ready")"
            return "\(primaryStatus) | \(secondaryStatus)"
        }
        
        return primaryStatus
    }
}
