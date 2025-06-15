//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import PhotosUI
import SwiftUI
import MLXLMCommon
import Foundation

#if canImport(UIKit)
import UIKit
typealias PlatformImage = UIImage
#elseif canImport(AppKit)
import AppKit
typealias PlatformImage = NSImage
#endif

struct EnhancedContentView: View {
    @State private var modelManager = ModelManager()
    @State private var selectedImages: [PhotosPickerItem] = []
    @State private var userInput = UserInput(messages: [])
    @State private var showingImagePicker = false
    @State private var showingModelSelector = false
    @State private var promptText: String = ""
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Model Selection Header
                modelSelectionHeader
                
                // Model Info
                modelInfoSection
                
                // Image Input Section
                imageInputSection
                
                // Text Input Section
                textInputSection
                
                // Generate Button
                generateButton
                
                // Output Section
                outputSection
                
                Spacer()
            }
            .padding()
            .navigationTitle("VLM Comparison")
            .task {
                await modelManager.loadCurrentModel()
            }
            .sheet(isPresented: $showingModelSelector) {
                modelSelectorSheet
            }
        }
    }
    
    private var modelSelectionHeader: some View {
        HStack {
            Text("Current Model:")
                .font(.headline)
            
            Spacer()
            
            Button(action: {
                showingModelSelector = true
            }) {
                HStack {
                    Text(modelManager.selectedModelType.displayName)
                        .foregroundColor(.primary)
                    Image(systemName: "chevron.down")
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(red: 0.95, green: 0.95, blue: 0.97))
                .cornerRadius(8)
            }
            .disabled(modelManager.isSwitchingModels || modelManager.running)
        }
    }
    
    private var modelInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if modelManager.isSwitchingModels {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text(modelManager.switchingProgress)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                Text(modelManager.modelInfo.isEmpty ? "Model not loaded" : modelManager.modelInfo)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                if !modelManager.promptTime.isEmpty {
                    Text("Processing Time: \(modelManager.promptTime)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    
    private var imageInputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Images")
                .font(.headline)
            
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 8) {
                    ForEach(userInput.images.indices, id: \.self) { index in
                        imageView(for: userInput.images[index])
                            .overlay(alignment: .topTrailing) {
                                removeImageButton(at: index)
                            }
                    }
                    
                    addImageButton
                }
                .padding(.horizontal)
            }
        }
        .onChange(of: selectedImages) { _, newValue in
            Task {
                await processSelectedImages(newValue)
            }
        }
    }
    
    private func removeImageButton(at index: Int) -> some View {
        Button {
            userInput.images.remove(at: index)
        } label: {
            Image(systemName: "xmark.circle.fill")
                .foregroundColor(.red)
                .background(Color.white, in: Circle())
        }
        .offset(x: 8, y: -8)
    }
    
    private var addImageButton: some View {
        PhotosPicker(selection: $selectedImages, maxSelectionCount: 4, matching: .images) {
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.gray, style: StrokeStyle(lineWidth: 2, dash: [5]))
                .frame(width: 80, height: 80)
                .overlay {
                    Image(systemName: "plus")
                        .foregroundColor(.gray)
                        .font(.title2)
                }
        }
    }
    
    private func processSelectedImages(_ items: [PhotosPickerItem]) async {
        for item in items {
            if let data = try? await item.loadTransferable(type: Data.self) {
                #if canImport(UIKit)
                if let uiImage = UIImage(data: data),
                   let ciImage = CIImage(image: uiImage) {
                    userInput.images.append(.ciImage(ciImage))
                }
                #elseif canImport(AppKit)
                if let nsImage = NSImage(data: data) {
                    // Convert NSImage to CIImage
                    if let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                        let ciImage = CIImage(cgImage: cgImage)
                        userInput.images.append(.ciImage(ciImage))
                    }
                }
                #endif
            }
        }
        selectedImages.removeAll()
    }
    
    private func imageView(for vlmImage: UserInput.Image) -> some View {
        Group {
            switch vlmImage {
            case .ciImage(let ciImage):
                // Convert CIImage to platform image for display
                if let cgImage = CIContext().createCGImage(ciImage, from: ciImage.extent) {
                    #if canImport(UIKit)
                    Image(uiImage: UIImage(cgImage: cgImage))
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                    #elseif canImport(AppKit)
                    Image(nsImage: NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height)))
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                    #endif
                } else {
                    // Fallback if conversion fails
                    Rectangle()
                        .fill(Color.gray)
                        .overlay {
                            Text("Image")
                                .foregroundColor(.white)
                        }
                }
            case .url(let url):
                AsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } placeholder: {
                    ProgressView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            case .array(_):
                // MLXArray image display not implemented
                Rectangle()
                    .fill(Color.orange)
                    .overlay {
                        Text("Array")
                            .foregroundColor(.white)
                            .font(.caption)
                    }
            }
        }
        .frame(width: 80, height: 80)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
    
    private var textInputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.headline)
            
            TextField("Enter your prompt here...", text: $promptText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...6)
        }
    }
    
    private var generateButton: some View {
        HStack {
            if modelManager.running {
                Button("Cancel") {
                    modelManager.cancel()
                }
                .buttonStyle(.bordered)
                
                Spacer()
                
                ProgressView()
                    .scaleEffect(0.8)
            } else {
                Button("Generate") {
                    Task {
                        let userPrompt = UserInput.Prompt.text(promptText)
                        
                        // Change from: userInput.prompt.text.isEmpty
                        // To: check the promptText string directly
                        if promptText.isEmpty {
                            // Handle empty prompt
                        } else {
                            await modelManager.generate(userInput)
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(userInput.images.isEmpty || promptText.isEmpty || modelManager.isSwitchingModels)
            }
        }
    }
    
    private var outputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Output")
                .font(.headline)
            
            ScrollView {
                Text(modelManager.output.isEmpty ? "Output will appear here..." : modelManager.output)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(red: 0.95, green: 0.95, blue: 0.97))
                    .cornerRadius(8)
            }
            .frame(minHeight: 100, maxHeight: 200)
        }
    }
    
    private var modelSelectorSheet: some View {
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
                            VStack(alignment: .leading) {
                                Text(modelType.displayName)
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                
                                Text(modelDescription(for: modelType))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            if modelType == modelManager.selectedModelType {
                                Image(systemName: "checkmark")
                                    .foregroundColor(.accentColor)
                            }
                        }
                    }
                    .disabled(modelManager.isSwitchingModels)
                }
            }
            .navigationTitle("Select Model")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        showingModelSelector = false
                    }
                }
            }
            #else
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        showingModelSelector = false
                    }
                }
            }
            #endif
        }
    }
    
    private func modelDescription(for modelType: ModelType) -> String {
        switch modelType {
        case .fastVLM:
            return "Uses Core ML vision encoder with MLX language model"
        case .smolVLM:
            return "Native MLX implementation of fine-tuned SmolVLM2"
        }
    }
}

#Preview {
    EnhancedContentView()
}
