//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI
import MLXLMCommon
import MLX

@Observable
@MainActor
class ModelManager {
    var selectedModelType: ModelType = .smolVLM
    var currentModel: any VLMModelProtocol
    
    // Model switching state
    var isSwitchingModels = false
    var switchingProgress: String = ""
    
    weak var speechManager: SpeechManager?
    
    private var lastMemoryWarning: Date?
    private let memoryWarningThreshold: TimeInterval = 5.0 // 5 seconds between warnings
    
    init() {
        #if !targetEnvironment(simulator)
        self.currentModel = ModelFactory.createModel(type: .smolVLM)
        
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handleMemoryWarning()
            }
        }
        #endif
        #else
        // On simulator, create a mock model that doesn't use GPU
        self.currentModel = ModelFactory.createModel(type: .smolVLM)
        print("Warning: Running on simulator - GPU functionality limited")
        #endif
    }
    
    func setSpeechManager(_ speechManager: SpeechManager) {
        self.speechManager = speechManager
        if let fastVLMModel = currentModel as? FastVLMModel {
            fastVLMModel.setSpeechManager(speechManager)
        } else if let smolVLMModel = currentModel as? SmolVLMModel {
            smolVLMModel.setSpeechManager(speechManager)
        }
    }
    
    private func handleMemoryWarning() {
        let now = Date()
        if let lastWarning = lastMemoryWarning,
           now.timeIntervalSince(lastWarning) < memoryWarningThreshold {
            return // Don't spam memory warnings
        }
        lastMemoryWarning = now
        
        print("[ModelManager] Memory warning received - triggering cleanup")
        
        // Cancel any running generation
        currentModel.cancel()
        
        // Force garbage collection
        Task {
            // Small delay to allow cancellation
            try? await Task.sleep(for: .milliseconds(100))
            
            // Trigger MLX cleanup
            MLX.GPU.clearCache()
        }
    }
    
    func switchModel(to modelType: ModelType) async {
        guard modelType != selectedModelType else { return }
        
        isSwitchingModels = true
        switchingProgress = "Switching to \(modelType.displayName)..."
        
        // Cancel any current generation
        currentModel.cancel()
        
        var waitCount = 0
        while currentModel.running && waitCount < 10 {
            try? await Task.sleep(for: .milliseconds(100))
            waitCount += 1
        }
        
        // This ensures clean memory management
        selectedModelType = modelType
        
        let oldModel = currentModel
        oldModel.cancel()
        
        // Clear MLX cache before loading new model
        MLX.GPU.clearCache()
        
        // Create new model
        currentModel = ModelFactory.createModel(type: modelType)
        
        if let speechManager = self.speechManager {
            if let fastVLMModel = currentModel as? FastVLMModel {
                fastVLMModel.setSpeechManager(speechManager)
            } else if let smolVLMModel = currentModel as? SmolVLMModel {
                smolVLMModel.setSpeechManager(speechManager)
            }
        }
        
        switchingProgress = "Loading \(modelType.displayName)..."
        await currentModel.load()
        
        Task {
            try? await Task.sleep(for: .milliseconds(500))
            MLX.GPU.clearCache()
        }
        
        isSwitchingModels = false
        switchingProgress = ""
    }
    
    func loadCurrentModel() async {
        await currentModel.load()
    }
    
    func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        if ProcessInfo.processInfo.physicalMemory > 0 {
            let usedMemory = getUsedMemory()
            let totalMemory = ProcessInfo.processInfo.physicalMemory
            let memoryUsagePercent = Double(usedMemory) / Double(totalMemory) * 100
            
            if memoryUsagePercent > 80 {
                print("[ModelManager] High memory usage: \(Int(memoryUsagePercent))% - triggering cleanup")
                MLX.GPU.clearCache()
            }
        }
        
        if running {
            print("[ModelManager] Cancelling existing generation to start new one")
            currentModel.cancel()
            
            var waitCount = 0
            while currentModel.running && waitCount < 20 { // Max 1 second wait
                try? await Task.sleep(for: .milliseconds(50))
                waitCount += 1
            }
            
            if currentModel.running {
                print("[ModelManager] Warning: Previous generation did not cancel cleanly")
            }
        }
        
        return await currentModel.generate(userInput)
    }
    
    func cancel() {
        currentModel.cancel()
    }
    
    private func getUsedMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
        
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return info.resident_size
        }
        return 0
    }
    
    // Computed properties for easy access
    var running: Bool {
        currentModel.running
    }
    
    var modelInfo: String {
        let primaryInfo = currentModel.modelInfo
        return primaryInfo.isEmpty ? "Loading \(selectedModelType.rawValue)..." : "\(selectedModelType.rawValue): \(primaryInfo)"
    }
    
    var output: String {
        currentModel.output
    }
    
    var promptTime: String {
        currentModel.promptTime
    }
    
    var allModelsStatus: String {
        return "\(selectedModelType.rawValue): \(running ? "Running" : "Ready")"
    }
    
    var memoryStatus: String {
        let usedMemory = getUsedMemory()
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let usedMB = usedMemory / 1024 / 1024
        let totalMB = totalMemory / 1024 / 1024
        let percentage = Double(usedMemory) / Double(totalMemory) * 100
        
        return "\(usedMB)MB / \(totalMB)MB (\(Int(percentage))%)"
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}
