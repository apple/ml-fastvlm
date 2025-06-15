//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXVLM
import MLXLMCommon
import MLXRandom
import Tokenizers
import CoreImage
import SwiftUI
import Hub
#if canImport(UIKit)
import UIKit
#endif

// MARK: - SmolVLM Error Types

public enum SmolVLMError: Error {
    case configurationError(String)
    case imageProcessingError(String)
    case tokenizationError(String)
    case modelLoadError(String)
}

// MARK: - SmolVLM Model Wrapper

@Observable
@MainActor
class SmolVLMModel: VLMModelProtocol {
    
    public var running = false
    public var modelInfo = "SmolVLM2 - Ready to load"
    public var output = ""
    public var promptTime: String = ""
    
    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }
    
    /// parameters controlling the output
    let maxTokens = 240
    
    /// update the display every N tokens
    let displayEveryNTokens = 4
    
    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?
    
    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }
    
    public var evaluationState = EvaluationState.idle
    
    // Model configuration - similar to FastVLM
    private var modelConfiguration: ModelConfiguration? = nil
    
    public init() {
        // SmolVLM initialization - set up model configuration
        setupModelConfiguration()
    }
    
    private func setupModelConfiguration() {
        do {
            let modelDir = try verifyModelBundle()
            
            // Create a model configuration that uses a known working model type
            // We'll try to use a Qwen2VL configuration as it's similar architecture
            // but override the model directory to point to our local SmolVLM model
            self.modelConfiguration = ModelConfiguration(directory: modelDir)
            
            // Try to patch the model type in the config.json to use a supported type
            try patchModelConfiguration(modelDir: modelDir)
            
        } catch {
            print("❌ Failed to setup SmolVLM model configuration: \(error)")
            self.modelInfo = "❌ Configuration Error: \(error.localizedDescription)"
        }
    }
    
    private func patchModelConfiguration(modelDir: URL) throws {
        let configURL = modelDir.appendingPathComponent("config.json")
        
        guard let configData = try? Data(contentsOf: configURL) else {
            throw SmolVLMError.configurationError("Failed to read config.json")
        }
        
        guard var config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw SmolVLMError.configurationError("Failed to parse config.json")
        }
        
        // Check the current model type
        let currentModelType = config["model_type"] as? String ?? "unknown"
        print("📋 Current model type: \(currentModelType)")
        
        if currentModelType == "smolvlm" {
            // The model type should be supported according to VLMModelFactory
            // Let's try using it as-is first
            print("✅ Using SmolVLM model type as-is")
        } else {
            print("⚠️  Unexpected model type: \(currentModelType)")
        }
    }
    
    private func verifyModelBundle() throws -> URL {
        guard let modelDir = Bundle.main.url(forResource: "SmolVLMModel", withExtension: nil) else {
            throw SmolVLMError.configurationError("SmolVLMModel bundle not found in app bundle")
        }
        
        let requiredFiles = [
            "config.json",
            "model.safetensors",
            "tokenizer.json"
        ]
        
        let optionalFiles = [
            "preprocessor_config.json",
            "processor_config.json"
        ]
        
        print("📁 SmolVLM Model Directory: \(modelDir.path)")
        print("🔍 Bundle identifier: \(Bundle.main.bundleIdentifier ?? "unknown")")
        
        // List all files in the SmolVLMModel directory
        do {
            let contents = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            print("📂 SmolVLMModel directory contents: \(contents.map { $0.lastPathComponent })")
        } catch {
            print("❌ Error listing SmolVLMModel directory contents: \(error)")
        }
        
        // Check required files
        for fileName in requiredFiles {
            let fileURL = modelDir.appendingPathComponent(fileName)
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            print("📄 \(fileName): \(exists ? "✅ Found" : "❌ Missing") at \(fileURL.path)")
            
            if !exists {
                throw SmolVLMError.configurationError("\(fileName) not found in SmolVLMModel bundle")
            }
        }
        
        // Check optional files
        for fileName in optionalFiles {
            let fileURL = modelDir.appendingPathComponent(fileName)
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            print("📄 \(fileName): \(exists ? "✅ Found" : "⚠️ Optional file missing") at \(fileURL.path)")
        }
        
        return modelDir
    }
    
    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            guard let modelConfiguration = self.modelConfiguration else {
                throw SmolVLMError.configurationError("Model configuration not set up")
            }
            
            // Set MLX memory limits
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
            
            // This may make things very slow when way over the limit
            let maxMetalMemory = Int(round(0.82 * Double(os_proc_available_memory())))
            MLX.GPU.set(memoryLimit: maxMetalMemory, relaxed: false)
            
            print("🚀 Loading SmolVLM2 model from: \(modelConfiguration.name)")
            
            // Try loading without Hub API first (local directory approach)
            do {
                let modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: modelConfiguration
                ) { progress in
                    Task { @MainActor in
                        self.modelInfo = "Loading SmolVLM2: \(Int(progress.fractionCompleted * 100))%"
                    }
                }
                
                // Get model info
                let numParams = await modelContainer.perform { context in
                    context.model.numParameters()
                }
                
                self.modelInfo = "✅ Loaded SmolVLM2 (\(numParams) parameters)"
                loadState = .loaded(modelContainer)
                return modelContainer
                
            } catch {
                print("❌ Failed to load SmolVLM with local directory approach: \(error)")
                throw SmolVLMError.configurationError("Failed to load SmolVLM model: \(error.localizedDescription)")
            }
            
        case .loaded(let modelContainer):
            return modelContainer
        }
    }
    
    public func load() async {
        do {
            _ = try await _load()
        } catch {
            self.modelInfo = "❌ Error loading SmolVLM: \(error.localizedDescription)"
            print("SmolVLM load error: \(error)")
        }
    }
    
    public func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        if let currentTask, running {
            return currentTask
        }
        
        running = true
        
        // Cancel any existing task
        currentTask?.cancel()
        
        // Create new task and store reference
        let task = Task {
            do {
                let modelContainer = try await _load()
                
                // each time you generate you will get something new
                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
                
                // Check if task was cancelled
                if Task.isCancelled { return }
                
                Task { @MainActor in
                    self.evaluationState = .processingPrompt
                    self.output = ""
                }
                
                let llmStart = Date()
                
                // Generate response using SmolVLM - same pattern as FastVLM
                let result = try await modelContainer.perform { context in
                    // Prepare the input using the model's processor
                    let input = try await context.processor.prepare(input: userInput)
                    
                    var seenFirstToken = false
                    
                    // Generate with proper parameters
                    let generationParameters = MLXLMCommon.GenerateParameters(
                        temperature: 0.3,
                        topP: 0.9
                    )
                    
                    return try MLXLMCommon.generate(
                        input: input,
                        parameters: generationParameters,
                        context: context
                    ) { tokens in
                        // Check if task was cancelled
                        if Task.isCancelled {
                            return .stop
                        }
                        
                        if !seenFirstToken {
                            seenFirstToken = true
                            
                            // produced first token, update the time to first token,
                            // the processing state and start displaying the text
                            let llmDuration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.evaluationState = .generatingResponse
                                self.output = text
                                self.promptTime = "\(Int(llmDuration * 1000)) ms"
                            }
                        }
                        
                        // Update the output as tokens are generated
                        if tokens.count % self.displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                if !Task.isCancelled {
                                    self.output = text
                                }
                            }
                        }
                        
                        if tokens.count >= self.maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                }
                
                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = result.output
                        self.promptTime += " | \(String(format: "%.1f", result.tokensPerSecond)) tok/s"
                    }
                }
                
            } catch SmolVLMError.configurationError(let message) {
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = "❌ SmolVLM Configuration Error: \(message)\n\nNote: This only affects SmolVLM model. FastVLM model should still work normally."
                    }
                }
            } catch SmolVLMError.imageProcessingError(let message) {
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = "❌ SmolVLM Image Processing Error: \(message)"
                    }
                }
            } catch {
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = "❌ SmolVLM Failed: \(error.localizedDescription)"
                    }
                }
            }
            
            Task { @MainActor in
                if self.evaluationState == .generatingResponse {
                    self.evaluationState = .idle
                }
                self.running = false
            }
        }
        
        currentTask = task
        return task
    }
    
    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        output = ""
        promptTime = ""
        evaluationState = .idle
    }
}
