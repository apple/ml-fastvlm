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
            
            // Try to patch the model configuration to use a supported type
            try patchModelConfiguration(modelDir: modelDir)
            
        } catch {
            print("[SmolVLM Debug] Failed to setup SmolVLM model configuration: \(error)")
            self.modelInfo = "Configuration Error: \(error.localizedDescription)"
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
        print("[SmolVLM Debug] Current model type: \(currentModelType)")
        
        if currentModelType == "smolvlm" {
            // The model type should be supported according to VLMModelFactory
            // Let's try using it as-is first
            print("[SmolVLM Debug] Using SmolVLM model type as-is")
        } else {
            print("[SmolVLM Debug] Unexpected model type: \(currentModelType)")
        }
        
        if let textConfig = config["text_config"] as? [String: Any] {
            print("[SmolVLM Debug] Text config vocab_size: \(textConfig["vocab_size"] ?? "unknown")")
            print("[SmolVLM Debug] Text config hidden_size: \(textConfig["hidden_size"] ?? "unknown")")
            print("[SmolVLM Debug] Text config num_hidden_layers: \(textConfig["num_hidden_layers"] ?? "unknown")")
        }
        
        if let visionConfig = config["vision_config"] as? [String: Any] {
            print("[SmolVLM Debug] Vision config image_size: \(visionConfig["image_size"] ?? "unknown")")
            print("[SmolVLM Debug] Vision config hidden_size: \(visionConfig["hidden_size"] ?? "unknown")")
        }
        
        print("[SmolVLM Debug] Image token ID: \(config["image_token_id"] ?? "unknown")")
        print("[SmolVLM Debug] Pad token ID: \(config["pad_token_id"] ?? "unknown")")
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
            "processor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt"
        ]
        
        print("[SmolVLM Debug] SmolVLM Model Directory: \(modelDir.path)")
        print("[SmolVLM Debug] Bundle identifier: \(Bundle.main.bundleIdentifier ?? "unknown")")
        
        // List all files in the SmolVLMModel directory
        do {
            let contents = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            print("[SmolVLM Debug] SmolVLMModel directory contents: \(contents.map { $0.lastPathComponent })")
        } catch {
            print("[SmolVLM Debug] Error listing SmolVLMModel directory contents: \(error)")
        }
        
        // Check required files
        for fileName in requiredFiles {
            let fileURL = modelDir.appendingPathComponent(fileName)
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            print("[SmolVLM Debug] \(fileName): \(exists ? "Found" : "Missing") at \(fileURL.path)")
            
            if !exists {
                throw SmolVLMError.configurationError("\(fileName) not found in SmolVLMModel bundle")
            }
        }
        
        // Check optional files
        for fileName in optionalFiles {
            let fileURL = modelDir.appendingPathComponent(fileName)
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            print("[SmolVLM Debug] \(fileName): \(exists ? "Found" : "Optional file missing") at \(fileURL.path)")
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
            
            // Check available memory - only use os_proc_available_memory on supported platforms
            #if os(iOS) || os(macOS)
            let maxMetalMemory: Int
            #if os(macOS)
            // On macOS, use a more conservative approach for Metal memory
            maxMetalMemory = Int(round(0.6 * Double(ProcessInfo.processInfo.physicalMemory)))
            #else
            // This may make things very slow when way over the limit
            maxMetalMemory = Int(round(0.82 * Double(os_proc_available_memory())))
            #endif
            MLX.GPU.set(memoryLimit: maxMetalMemory, relaxed: false)
            print("[SmolVLM Debug] Set Metal memory limit to: \(maxMetalMemory / 1024 / 1024) MB")
            #endif
            
            print("[SmolVLM Debug] Loading SmolVLM2 model from: \(modelConfiguration.name)")
            
            // Try loading without Hub API first (local directory approach)
            do {
                let modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: modelConfiguration
                ) { progress in
                    Task { @MainActor in
                        self.modelInfo = "Loading SmolVLM2: \(Int(progress.fractionCompleted * 100))%"
                    }
                }
                
                self.modelInfo = "Loaded SmolVLM2"
                loadState = .loaded(modelContainer)
                print("[SmolVLM Debug] Model loaded successfully")
                return modelContainer
                
            } catch {
                print("[SmolVLM Debug] Failed to load SmolVLM with local directory approach: \(error)")
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
            self.modelInfo = "Error loading SmolVLM: \(error.localizedDescription)"
            print("[SmolVLM Debug] SmolVLM load error: \(error)")
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
                
                print("[SmolVLM Debug] === Generation Started ===")
                print("[SmolVLM Debug] User prompt: \(userInput.prompt)")
                print("[SmolVLM Debug] Number of images: \(userInput.images.count)")
                
                // Generate response using SmolVLM - same pattern as FastVLM
                let result = try await modelContainer.perform { context in
                    print("[SmolVLM Debug] Preparing input with processor...")
                    
                    // Prepare the input using the model's processor
                    let input = try await context.processor.prepare(input: userInput)
                    
                    print("[SmolVLM Debug] Input prepared successfully")
                    print("[SmolVLM Debug] Input type: \(type(of: input))")
                    
                    var seenFirstToken = false
                    var generatedTokens: [Int] = []
                    
                    // Generate with proper parameters - ADD: More conservative parameters for debugging
                    let generationParameters = MLXLMCommon.GenerateParameters(
                        temperature: 0.1,  // Lower temperature for more deterministic output
                        topP: 0.95,        // Slightly higher top-p
                        repetitionPenalty: 1.05  // Small repetition penalty
                    )
                    
                    print("[SmolVLM Debug] Generation parameters: temp=\(generationParameters.temperature ?? 0), topP=\(generationParameters.topP ?? 0)")
                    
                    return try MLXLMCommon.generate(
                        input: input,
                        parameters: generationParameters,
                        context: context
                    ) { tokens in
                        // Check if task was cancelled
                        if Task.isCancelled {
                            return .stop
                        }
                        
                        let newTokens = Array(tokens.suffix(tokens.count - generatedTokens.count))
                        if !newTokens.isEmpty {
                            print("[SmolVLM Debug] New tokens: \(newTokens)")
                            let newText = context.tokenizer.decode(tokens: newTokens)
                            print("[SmolVLM Debug] New text: '\(newText)'")
                        }
                        generatedTokens = tokens
                        
                        if !seenFirstToken {
                            seenFirstToken = true
                            
                            // produced first token, update the time to first token,
                            // the processing state and start displaying the text
                            let llmDuration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            print("[SmolVLM Debug] First token generated in \(Int(llmDuration * 1000)) ms")
                            print("[SmolVLM Debug] Current text: '\(text)'")
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
                        
                        let currentText = context.tokenizer.decode(tokens: tokens)
                        
                        // Stop if generating repetitive content
                        if currentText.count > 50 {
                            let lastPart = String(currentText.suffix(20))
                            let beforeLastPart = String(currentText.dropLast(20).suffix(20))
                            if lastPart == beforeLastPart {
                                print("[SmolVLM Debug] Detected repetitive pattern, stopping generation")
                                return .stop
                            }
                        }
                        
                        if tokens.count >= self.maxTokens {
                            print("[SmolVLM Debug] Reached max tokens (\(self.maxTokens)), stopping")
                            return .stop
                        } else {
                            return .more
                        }
                    }
                }
                
                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    print("[SmolVLM Debug] === Generation Completed ===")
                    print("[SmolVLM Debug] Final output: '\(result.output)'")
                    print("[SmolVLM Debug] Tokens per second: \(result.tokensPerSecond)")
                    
                    Task { @MainActor in
                        self.output = result.output
                        self.promptTime += " | \(String(format: "%.1f", result.tokensPerSecond)) tok/s"
                    }
                }
                
            } catch SmolVLMError.configurationError(let message) {
                print("[SmolVLM Debug] Configuration error: \(message)")
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = "SmolVLM Configuration Error: \(message)\n\nNote: This only affects SmolVLM model. FastVLM model should still work normally."
                    }
                }
            } catch SmolVLMError.imageProcessingError(let message) {
                print("[SmolVLM Debug] Image processing error: \(message)")
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = "SmolVLM Image Processing Error: \(message)"
                    }
                }
            } catch {
                print("[SmolVLM Debug] Generation error: \(error)")
                if !Task.isCancelled {
                    Task { @MainActor in
                        self.output = "SmolVLM Failed: \(error.localizedDescription)"
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
        print("[SmolVLM Debug] Generation cancelled")
    }
}
