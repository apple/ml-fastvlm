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
    
    private var isRecovering = false
    private var recoveryAttempts = 0
    private let maxRecoveryAttempts = 3
    
    init() {
        #if !targetEnvironment(simulator)
        self.currentModel = ModelFactory.createModel(type: .smolVLM)
        
        setupNotificationObservers()
        #else
        // On simulator, create a mock model that doesn't use GPU
        self.currentModel = ModelFactory.createModel(type: .smolVLM)
        print("Warning: Running on simulator - GPU functionality limited")
        #endif
    }
    
    private func setupNotificationObservers() {
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
        
        NotificationCenter.default.addObserver(
            forName: .appEnteredBackground,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handleAppEnteredBackground()
            }
        }
        
        NotificationCenter.default.addObserver(
            forName: .memoryPressureDetected,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handleMemoryPressure()
            }
        }
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
        
        // Clear output to free memory
        currentModel.output = ""
        
        // Force garbage collection
        Task {
            // Small delay to allow cancellation
            try? await Task.sleep(for: .milliseconds(100))
            
            // Trigger MLX cleanup
            MLX.GPU.clearCache()
            
            // If memory pressure is severe, consider reloading model
            if getUsedMemory() > 7000 * 1024 * 1024 { // 7GB
                await performEmergencyRecovery()
            }
        }
    }
    
    private func handleAppEnteredBackground() {
        print("[ModelManager] App entered background - performing cleanup")
        
        // Cancel any running generation
        currentModel.cancel()
        
        // Clear output to save memory
        currentModel.output = ""
        
        // Clear MLX cache
        MLX.GPU.clearCache()
    }
    
    private func handleMemoryPressure() {
        print("[ModelManager] Memory pressure detected - performing aggressive cleanup")
        
        // Cancel any running generation
        currentModel.cancel()
        
        // Clear output
        currentModel.output = ""
        
        // Clear MLX cache
        MLX.GPU.clearCache()
        
        // If pressure is severe, perform emergency recovery
        Task {
            let memoryUsage = getUsedMemory()
            if memoryUsage > 6500 * 1024 * 1024 { // 6.5GB
                await performEmergencyRecovery()
            }
        }
    }
    
    private func performEmergencyRecovery() async {
        guard !isRecovering else { return }
        guard recoveryAttempts < maxRecoveryAttempts else {
            print("[ModelManager] Max recovery attempts reached - giving up")
            return
        }
        
        isRecovering = true
        recoveryAttempts += 1
        
        print("[ModelManager] Performing emergency recovery (attempt \(recoveryAttempts))")
        
        // Force cancel everything
        currentModel.cancel()
        
        // Wait for cancellation
        var waitCount = 0
        while currentModel.running && waitCount < 20 {
            try? await Task.sleep(for: .milliseconds(100))
            waitCount += 1
        }
        
        // Clear all state
        currentModel.output = ""
        
        // Aggressive cache clearing
        MLX.GPU.clearCache()
        
        // Small delay
        try? await Task.sleep(for: .milliseconds(500))
        
        // Try to reload the model
        do {
            await currentModel.load()
            print("[ModelManager] Emergency recovery successful")
            recoveryAttempts = 0 // Reset on success
        } catch {
            print("[ModelManager] Emergency recovery failed: \(error)")
            
            // If recovery failed, try switching to a different model
            if recoveryAttempts >= 2 {
                let alternativeModel: ModelType = selectedModelType == .fastVLM ? .smolVLM : .fastVLM
                print("[ModelManager] Switching to alternative model: \(alternativeModel)")
                await switchModel(to: alternativeModel)
            }
        }
        
        isRecovering = false
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
        
        // Small delay for cleanup
        try? await Task.sleep(for: .milliseconds(200))
        
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
        
        do {
            await currentModel.load()
            recoveryAttempts = 0 // Reset on successful load
        } catch {
            print("[ModelManager] Failed to load model: \(error)")
            // Could attempt recovery here
        }
        
        Task {
            try? await Task.sleep(for: .milliseconds(500))
            MLX.GPU.clearCache()
        }
        
        isSwitchingModels = false
        switchingProgress = ""
    }
    
    func loadCurrentModel() async {
        do {
            await currentModel.load()
        } catch {
            print("[ModelManager] Failed to load current model: \(error)")
            // Attempt recovery
            await performEmergencyRecovery()
        }
    }
    
    func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        // Check if we're in a recovery state
        if isRecovering {
            print("[ModelManager] Currently recovering - skipping generation")
            return Task { }
        }
        
        if ProcessInfo.processInfo.physicalMemory > 0 {
            let usedMemory = getUsedMemory()
            let totalMemory = ProcessInfo.processInfo.physicalMemory
            let memoryUsagePercent = Double(usedMemory) / Double(totalMemory) * 100
            
            if memoryUsagePercent > 80 {
                print("[ModelManager] High memory usage: \(Int(memoryUsagePercent))% - triggering cleanup")
                MLX.GPU.clearCache()
                
                // If memory is critically high, perform emergency recovery
                if memoryUsagePercent > 90 {
                    Task {
                        await performEmergencyRecovery()
                    }
                    return Task { }
                }
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
                // Force recovery if cancellation fails
                Task {
                    await performEmergencyRecovery()
                }
                return Task { }
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
        let recoveryInfo = isRecovering ? " (Recovering)" : ""
        return primaryInfo.isEmpty ? "Loading \(selectedModelType.rawValue)...\(recoveryInfo)" : "\(selectedModelType.rawValue): \(primaryInfo)\(recoveryInfo)"
    }
    
    var output: String {
        currentModel.output
    }
    
    var promptTime: String {
        currentModel.promptTime
    }
    
    var allModelsStatus: String {
        let status = running ? "Running" : (isRecovering ? "Recovering" : "Ready")
        return "\(selectedModelType.rawValue): \(status)"
    }
    
    var memoryStatus: String {
        let usedMemory = getUsedMemory()
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let usedMB = usedMemory / 1024 / 1024
        let totalMB = totalMemory / 1024 / 1024
        let percentage = Double(usedMemory) / Double(totalMemory) * 100
        
        let pressureInfo = isRecovering ? " (Recovering)" : ""
        return "\(usedMB)MB / \(totalMB)MB (\(Int(percentage))%)\(pressureInfo)"
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}
