//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import MLXLMCommon
import SwiftUI
import Video

#if canImport(UIKit)
import UIKit
#endif

// support swift 6
extension CVImageBuffer: @unchecked @retroactive Sendable {}
extension CMSampleBuffer: @unchecked @retroactive Sendable {}

// delay between frames -- controls the frame rate of the updates
let FRAME_DELAY = Duration.milliseconds(1)

struct ContentView: View {
    @State private var camera = CameraController()
    @State private var modelManager = ModelManager()

    /// stream of frames -> VideoFrameView, see distributeVideoFrames
    @State private var framesToDisplay: AsyncStream<CVImageBuffer>?

    @State private var prompt = "Describe the image in English."
    @State private var promptSuffix = "Output should be brief, about 15 words or less."

    @State private var isShowingInfo: Bool = false
    @State private var showingModelSelector = false

    @State private var selectedCameraType: CameraType = .single
    @State private var isEditingPrompt: Bool = false
    
    var statusTextColor: Color {
        guard let fastVLMModel = modelManager.currentModel as? FastVLMModel else {
            return .white
        }
        return fastVLMModel.evaluationState == .processingPrompt ? .black : .white
    }
    
    var statusBackgroundColor: Color {
        guard let fastVLMModel = modelManager.currentModel as? FastVLMModel else {
            return modelManager.running ? .green : .gray
        }
        
        switch fastVLMModel.evaluationState {
        case .idle:
            return .gray
        case .generatingResponse:
            return .green
        case .processingPrompt:
            return .yellow
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                cameraSection
                promptSections
                responseSection
            }
            #if os(iOS)
            .listSectionSpacing(0)
            #endif
            .task {
                camera.start()
            }
            .task {
                await modelManager.loadCurrentModel()
            }
            .onAppear {
                #if canImport(UIKit)
                UIApplication.shared.isIdleTimerDisabled = true
                #endif
            }
            .onDisappear {
                #if canImport(UIKit)
                UIApplication.shared.isIdleTimerDisabled = false
                #endif
            }
            .task {
                if Task.isCancelled {
                    return
                }
                await distributeVideoFrames()
            }
            .modifier(NavigationBarModifier(
                modelManager: modelManager,
                isEditingPrompt: $isEditingPrompt,
                showingModelSelector: $showingModelSelector,
                isShowingInfo: $isShowingInfo,
                prompt: $prompt,
                promptSuffix: $promptSuffix
            ))
            .sheet(isPresented: $isShowingInfo) {
                InfoView()
            }
            .sheet(isPresented: $showingModelSelector) {
                modelSelectorSheet
            }
        }
    }
    
    private var cameraSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 10.0) {
                Picker("Camera Type", selection: $selectedCameraType) {
                    ForEach(CameraType.allCases, id: \.self) { cameraType in
                        Text(cameraType.rawValue.capitalized).tag(cameraType)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .onChange(of: selectedCameraType) { _, _ in
                    modelManager.cancel()
                }

                if let framesToDisplay {
                    videoFrameView(framesToDisplay)
                }
            }
        }
        .listRowInsets(EdgeInsets())
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
    }
    
    private func videoFrameView(_ frames: AsyncStream<CVImageBuffer>) -> some View {
        VideoFrameView(
            frames: frames,
            cameraType: selectedCameraType,
            action: { frame in
                processSingleFrame(frame)
            })
            .aspectRatio(4/3, contentMode: .fit)
            .overlay(alignment: .top) {
                timeToFirstTokenOverlay
            }
            .overlay(alignment: .topTrailing) {
                cameraControlsOverlay
            }
            .overlay(alignment: .bottom) {
                statusOverlay
            }
    }
    
    private var timeToFirstTokenOverlay: some View {
        Group {
            if !modelManager.promptTime.isEmpty {
                Text("TTFT \(modelManager.promptTime)")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .monospaced()
                    .padding(.vertical, 4.0)
                    .padding(.horizontal, 6.0)
                    .background(alignment: .center) {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.black.opacity(0.6))
                    }
                    .padding(.top)
            }
        }
    }
    
    private var cameraControlsOverlay: some View {
        Group {
            if let device = camera.device {
                CameraControlsView(
                    backCamera: $camera.backCamera,
                    device: .constant(device),
                    devices: $camera.devices)
                .padding()
            }
        }
    }
    
    private var statusOverlay: some View {
        Group {
            if selectedCameraType == .continuous {
                statusIndicatorView
                    .foregroundStyle(statusTextColor)
                    .font(.caption)
                    .bold()
                    .padding(.vertical, 6.0)
                    .padding(.horizontal, 8.0)
                    .background(statusBackgroundColor)
                    .clipShape(.capsule)
                    .padding(.bottom)
            }
        }
    }
    
    private var responseSection: some View {
        Section {
            if modelManager.output.isEmpty && modelManager.running {
                ProgressView()
                    .controlSize(.large)
                    .frame(maxWidth: .infinity)
            } else {
                ScrollView {
                    Text(modelManager.output)
                        .foregroundStyle(isEditingPrompt ? .secondary : .primary)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 50.0, maxHeight: 200.0)
            }
        } header: {
            Text("Response")
        }
    }
    
    @ViewBuilder
    private var statusIndicatorView: some View {
        if let fastVLMModel = modelManager.currentModel as? FastVLMModel, 
           fastVLMModel.evaluationState == .processingPrompt {
            HStack {
                ProgressView()
                    .tint(statusTextColor)
                    .controlSize(.small)
                Text(fastVLMModel.evaluationState.rawValue)
            }
        } else if let fastVLMModel = modelManager.currentModel as? FastVLMModel, 
                  fastVLMModel.evaluationState == .idle {
            HStack(spacing: 6.0) {
                Image(systemName: "clock.fill")
                    .font(.caption)
                Text(fastVLMModel.evaluationState.rawValue)
            }
        } else {
            HStack(spacing: 6.0) {
                Image(systemName: "lightbulb.fill")
                    .font(.caption)
                Text(modelManager.running ? "Generating Response" : "Idle")
            }
        }
    }

    var promptSummary: some View {
        Section("Prompt") {
            VStack(alignment: .leading, spacing: 4.0) {
                let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedPrompt.isEmpty {
                    Text(trimmedPrompt)
                        .foregroundStyle(.secondary)
                }

                let trimmedSuffix = promptSuffix.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedSuffix.isEmpty {
                    Text(trimmedSuffix)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    var promptForm: some View {
        Group {
            Section("Prompt") {
                TextEditor(text: $prompt)
                    .frame(minHeight: 38)
            }

            Section("Prompt Suffix") {
                TextEditor(text: $promptSuffix)
                    .frame(minHeight: 38)
            }
        }
    }

    var promptSections: some View {
        Group {
            if isEditingPrompt {
                promptForm
            } else {
                promptSummary
            }
        }
    }

    func analyzeVideoFrames(_ frames: AsyncStream<CVImageBuffer>) async {
        for await frame in frames {
            let userInput = UserInput(
                prompt: .text("\(prompt) \(promptSuffix)"),
                images: [.ciImage(CIImage(cvPixelBuffer: frame))]
            )
            
            // generate output for a frame and wait for generation to complete
            let t = await modelManager.generate(userInput)
            _ = await t.result

            do {
                try await Task.sleep(for: FRAME_DELAY)
            } catch { return }
        }
    }

    func distributeVideoFrames() async {
        // attach a stream to the camera -- this code will read this
        let frames = AsyncStream<CMSampleBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            camera.attach(continuation: $0)
        }

        let (framesToDisplay, framesToDisplayContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        self.framesToDisplay = framesToDisplay

        // Only create analysis stream if in continuous mode
        let (framesToAnalyze, framesToAnalyzeContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )

        // set up structured tasks (important -- this means the child tasks
        // are cancelled when the parent is cancelled)
        async let distributeFrames: () = {
            for await sampleBuffer in frames {
                if let frame = sampleBuffer.imageBuffer {
                    framesToDisplayContinuation.yield(frame)
                    // Only send frames for analysis in continuous mode
                    if await selectedCameraType == .continuous {
                        framesToAnalyzeContinuation.yield(frame)
                    }
                }
            }

            // detach from the camera controller and feed to the video view
            await MainActor.run {
                self.framesToDisplay = nil
                self.camera.detatch()
            }

            framesToDisplayContinuation.finish()
            framesToAnalyzeContinuation.finish()
        }()

        // Only analyze frames if in continuous mode
        if selectedCameraType == .continuous {
            async let analyze: () = analyzeVideoFrames(framesToAnalyze)
            await distributeFrames
            await analyze
        } else {
            await distributeFrames
        }
    }

    /// Perform VLM inference on a single frame.
    /// - Parameter frame: The frame to analyze.
    func processSingleFrame(_ frame: CVImageBuffer) {
        // Reset Response UI (show spinner)
        Task { @MainActor in
            modelManager.currentModel.output = ""
        }

        // Construct request to model
        let userInput = UserInput(
            prompt: .text("\(prompt) \(promptSuffix)"),
            images: [.ciImage(CIImage(cvPixelBuffer: frame))]
        )

        // Post request to VLM
        Task {
            await modelManager.generate(userInput)
        }
    }
}

// MARK: - Navigation Bar Modifier

struct NavigationBarModifier: ViewModifier {
    let modelManager: ModelManager
    @Binding var isEditingPrompt: Bool
    @Binding var showingModelSelector: Bool
    @Binding var isShowingInfo: Bool
    @Binding var prompt: String
    @Binding var promptSuffix: String
    
    func body(content: Content) -> some View {
        content
            .navigationBarBackButtonHidden(false)
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Button {
                        showingModelSelector.toggle()
                    } label: {
                        HStack {
                            Image(systemName: "cpu")
                            Text(modelManager.selectedModelType.rawValue)
                                .font(.headline)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.primary)
                    }
                    .disabled(modelManager.isSwitchingModels || modelManager.running)
                }

                #if os(iOS)
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        isShowingInfo.toggle()
                    } label: {
                        Image(systemName: "info.circle")
                    }
                }
                #else
                ToolbarItem(placement: .navigation) {
                    Button {
                        isShowingInfo.toggle()
                    } label: {
                        Image(systemName: "info.circle")
                    }
                }
                #endif

                ToolbarItem(placement: .primaryAction) {
                    if isEditingPrompt {
                        Button {
                            isEditingPrompt.toggle()
                        } label: {
                            Text("Done")
                                .fontWeight(.bold)
                        }
                    } else {
                        Menu {
                            Button("Custom System Prompt") {
                                prompt = "You are an AI assistant specialized in describing visual scenes to blind and visually impaired individuals. Your purpose is to help users better understand their surroundings while navigating outdoor environments.\n\nThe user is walking outdoors and uses a white cane to follow physical cues such as walls, curbs, and transitions between different surfaces like grass, bike paths, and sidewalks. The user may ask you for information when they are unsure about what is around them, when they are unsure how or whether to proceed, or when their white cane does not provide enough information.\n\nYou must not provide the following types of information, even if the user asks for them:\n\n- Distances to objects or features in the environment (e.g., in meters, steps, or similar terms)\n\n- The color of traffic lights (e.g., \"The traffic light is red\" or \"green\")\n\n- Any assessment of whether it is safe or unsafe to cross a street\n\n- Whether a vehicle is moving or stationary\n\nIf the user asks about any of these, always respond with:\n\"I cannot answer that, please ask something else.\"\n\nInteraction guidelines:\n\n- Always refer only to the latest image provided\n\n- Respond promptly, briefly, and clearly\n\n- Provide only significant and relevant information\n\n- Use informal, polite, and patient language\n\n- If the image does not contain what the user is asking about, reply:\n\"I cannot see what you are asking about in the image.\"\n\n- Do not mention materials, surfaces, or colors unless the user explicitly asks\n\n- Always end your response with an invitation for follow-up, such as:\n\"Feel free to ask me another question.\"\n\n- When describing objects, explain their spatial arrangement and their position relative to the user using clock directions (e.g., \"The bench is at 2 o'clock.\")\n\nTypes of questions the user might ask:\n\nGeneral scene questions (e.g., \"Please describe the area in front of me\")\n→ Respond with one concise sentence summarizing the scene.\n\nSpecific questions about objects or navigation (e.g., \"What is in front of me?\", \"Where is the bench?\", \"What number is this bus?\")\n→ Name the visible objects, describe how they are arranged, and specify their location using clock directions relative to the user's position."
                                promptSuffix = ""
                            }
                            Button("Describe image") {
                                prompt = "Describe the image in English. Output should be brief, about 15 words or less."
                                promptSuffix = ""
                            }
                            Button("Read text") {
                                prompt = "What is written in this image?"
                                promptSuffix = "Output only the text in the image."
                            }
                            Button("Customize...") {
                                isEditingPrompt.toggle()
                            }
                        } label: { 
                            Text("Prompts") 
                        }
                    }
                }
            }
    }
}

#Preview {
    ContentView()
}

// MARK: - Model Selector Sheet
    
private extension ContentView {
    var modelSelectorSheet: some View {
        NavigationView {
            List {
                ForEach(ModelType.allCases, id: \.self) { modelType in
                    Button(action: {
                        Task {
                            showingModelSelector = false
                            await modelManager.switchModel(to: modelType)
                        }
                    }) {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(modelType.displayName)
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                
                                Text(modelDescription(for: modelType))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.leading)
                            }
                            
                            Spacer()
                            
                            if modelType == modelManager.selectedModelType {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.accentColor)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .disabled(modelManager.isSwitchingModels)
                }
                
                if modelManager.isSwitchingModels {
                    Section {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text(modelManager.switchingProgress)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 8)
                    }
                }
                
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Current Model Info")
                            .font(.headline)
                        
                        Text(modelManager.modelInfo.isEmpty ? "No model loaded" : modelManager.modelInfo)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        if !modelManager.promptTime.isEmpty {
                            Text("Last Processing Time: \(modelManager.promptTime)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Select Model")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        showingModelSelector = false
                    }
                }
            }
        }
    }
    
    func modelDescription(for modelType: ModelType) -> String {
        switch modelType {
        case .fastVLM:
            return "Uses Core ML vision encoder with MLX language model. Optimized for real-time inference."
        case .smolVLM:
            return "Native MLX implementation of fine-tuned SmolVLM2. Better scene understanding for construction sites."
        }
    }
}
