//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

struct InfoView: View {
    @Environment(\.dismiss) var dismiss

    let paragraph1 = "**SceneDescriptor** is a versatile iOS app that supports multiple Vision-Language models for scene understanding and description. The app can utilize both **FastVLM** and a specialized **SmolVLM2** model to provide comprehensive visual analysis."
    let paragraph2 = "**FastVLM¹** is a new family of Vision-Language models that makes use of **FastViTHD**, a hierarchical hybrid vision encoder that produces small number of high quality tokens at low latitudes, resulting in significantly faster time-to-first-token (TTFT). FastVLM utilizes Qwen2-Instruct LLMs without additional safety tuning, so please exercise caution when modifying the prompt."
    let paragraph3 = "Additionally, the app features a fine-tuned **SmolVLM2** model specifically optimized for providing scene descriptions to blind and visually impaired users. This model has been converted to **MLX format** for efficient on-device inference and is trained to provide helpful, contextual descriptions of visual environments for accessibility use cases."
    let paragraph4 = "The app includes text-to-speech functionality for hands-free operation, real-time camera analysis, and supports both single-shot and continuous video analysis modes. All processing happens locally on your device for privacy and immediate response times."
    let footer = "1. **FastVLM: Efficient Vision Encoding for Vision Language Models.** (CVPR 2025) Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari\n\nBuilt with MLX framework for Apple Silicon. SmolVLM2 fine-tuning and accessibility optimization by the SceneDescriptor team."

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20.0) {
                // I'm not going to lie, this doesn't make sense...
                // Wrapping `String`s with `.init()` turns them into `LocalizedStringKey`s
                // which gives us all of the fun Markdown formatting while retaining the
                // ability to use `String` variables. ¯\_(ツ)_/¯
                Text("\(.init(paragraph1))\n\n\(.init(paragraph2))\n\n\(.init(paragraph3))\n\n\(.init(paragraph4))\n\n")
                    .font(.body)

                Spacer()

                Text(.init(footer))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .textSelection(.enabled)
            .navigationTitle("About SceneDescriptor")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                #if os(iOS)
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle")
                            .resizable()
                            .frame(width: 25, height: 25)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                #elseif os(macOS)
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                    .buttonStyle(.bordered)
                }
                #endif
            }
        }
    }
}

#Preview {
    InfoView()
}
