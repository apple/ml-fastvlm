//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import FastVLM
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM

@Observable
@MainActor
class FastVLMModel: VLMModelProtocol {

    public var running = false
    public var modelInfo = ""
    public var output = ""
    public var promptTime: String = ""

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    private let modelConfiguration = FastVLM.modelConfiguration

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.1)
    let maxTokens = 300

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 2

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle

    public init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
    }

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            MLX.GPU.set(cacheLimit: 100 * 1024 * 1024) // 100MB cache limit (same as SmolVLM)

            #if os(iOS) || os(macOS)
            let maxMetalMemory: Int
            #if os(macOS)
            // Use same conservative memory strategy as SmolVLM for fair comparison
            let totalMemory = ProcessInfo.processInfo.physicalMemory
            let currentUsage = 1024 * 1024 * 1024 // 1GB current usage estimate
            let availableMemory = totalMemory - currentUsage
            maxMetalMemory = min(3 * 1024 * 1024 * 1024, Int(round(0.4 * Double(availableMemory)))) // Max #GB or 40% of available (same as SmolVLM)
            #else
            maxMetalMemory = min(3 * 1024 * 1024 * 1024, Int(round(0.7 * Double(os_proc_available_memory())))) // Max 2GB or 70% available
            #endif
            MLX.GPU.set(memoryLimit: maxMetalMemory, relaxed: true)
            print("[FastVLM Debug] Set Metal memory limit to: \(maxMetalMemory / 1024 / 1024) MB (optimized for 8GB iOS device)")
            #endif

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo = "Loaded"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    public func load() async {
        do {
            _ = try await _load()
        } catch {
            self.modelInfo = "Error loading model: \(error)"
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

                let result = try await modelContainer.perform { context in
                    // Measure the time it takes to prepare the input
                    
                    Task { @MainActor in
                        evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
                    let input = try await context.processor.prepare(input: userInput)
                    
                    print("[FastVLM Debug] === Input Preparation ===")
                    let textInput = input.text
                    print("[FastVLM Debug] Text tokens shape: \(textInput.tokens.shape)")
                    print("[FastVLM Debug] First 20 tokens: \(Array(textInput.tokens.asArray(Int32.self).prefix(20)))")
                    
                    // Decode first few tokens to see what text we're sending
                    let firstTokens = Array(textInput.tokens.flattened().asArray(Int32.self).prefix(50))
                    let decodedText = context.tokenizer.decode(tokens: firstTokens.map(Int.init))
                    print("[FastVLM Debug] Decoded input text: '\(decodedText)'")
                    
                    if input.image != nil {
                        print("[FastVLM Debug] Image input present: \(input.image!.pixels.shape)")
                    }
                    print("[FastVLM Debug] === End Input Debug ===")

                    var seenFirstToken = false

                    // FastVLM generates the output
                    let result = try MLXLMCommon.generate(
                        input: input, parameters: generateParameters, context: context
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
                                evaluationState = .generatingResponse
                                self.output = text
                                self.promptTime = "\(Int(llmDuration * 1000)) ms"
                            }
                        }

                        // Show the text in the view as it generates
                        if tokens.count % displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.output = text
                            }
                        }

                        if tokens.count >= maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                    
                    // Return the duration of the LLM and the result
                    return result
                }
                
                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    print("[FastVLM Debug] === Generation Completed ===")
                    print("[FastVLM Debug] Final output: '\(result.output)'")
                    print("[FastVLM Debug] Tokens per second: \(result.tokensPerSecond)")
                    
                    Task { @MainActor in
                        self.output = result.output
                        self.promptTime += " | \(String(format: "%.1f", result.tokensPerSecond)) tok/s"
                    }
                }

            } catch {
                if !Task.isCancelled {
                    output = "Failed: \(error)"
                }
            }

            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }

            running = false
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
    }
}
