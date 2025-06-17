import SwiftUI
import AVFoundation

struct SpeechSettingsView: View {
    @ObservedObject var speechManager: SpeechManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var selectedVoiceIndex: Int = 0
    @State private var speechRate: Double = 0.5
    @State private var testText: String = "This is a test of the text-to-speech functionality."
    
    private let availableVoices: [AVSpeechSynthesisVoice]
    
    init(speechManager: SpeechManager) {
        self.speechManager = speechManager
        self.availableVoices = speechManager.getAvailableVoices()
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section("Auto-Read Settings") {
                    Toggle("Automatically read continuous responses", isOn: $speechManager.autoReadResponses)
                    
                    Text("When enabled, vision model responses will be spoken automatically without needing to tap the speaker button.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Section("Voice Settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Speech Rate")
                        HStack {
                            Text("Slow")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: $speechRate, in: 0.1...1.0, step: 0.1)
                            Text("Fast")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Text("Rate: \(speechRateDescription)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .onChange(of: speechRate) { _, newValue in
                        let selectedVoice = availableVoices.indices.contains(selectedVoiceIndex) ? availableVoices[selectedVoiceIndex] : nil
                        speechManager.updatePreferences(voice: selectedVoice, rate: Float(newValue))
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("Voice", selection: $selectedVoiceIndex) {
                            ForEach(availableVoices.indices, id: \.self) { index in
                                Text(voiceDisplayName(for: availableVoices[index]))
                                    .tag(index)
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: selectedVoiceIndex) { _, newValue in
                            let selectedVoice = availableVoices.indices.contains(newValue) ? availableVoices[newValue] : nil
                            speechManager.updatePreferences(voice: selectedVoice, rate: Float(speechRate))
                        }
                    }
                }
                
                Section("Test Speech") {
                    VStack(alignment: .leading, spacing: 12) {
                        TextEditor(text: $testText)
                            .frame(minHeight: 80)
                        
                        HStack(spacing: 12) {
                            Button("Test Voice") {
                                let selectedVoice = availableVoices[selectedVoiceIndex]
                                speechManager.speak(testText, rate: Float(speechRate), voice: selectedVoice)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(testText.isEmpty)
                            
                            if speechManager.isSpeaking {
                                Button("Stop") {
                                    speechManager.stop()
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Speech Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                loadCurrentSettings()
            }
        }
    }
    
    private var speechRateDescription: String {
        let formattedRate = String(format: "%.1f", speechRate)
        switch speechRate {
        case 0.1...0.3:
            return "Very Slow (\(formattedRate))"
        case 0.4...0.4:
            return "Slow (\(formattedRate))"
        case 0.5...0.5:
            return "Normal (\(formattedRate))"
        case 0.6...0.7:
            return "Fast (\(formattedRate))"
        case 0.8...1.0:
            return "Very Fast (\(formattedRate))"
        default:
            return formattedRate
        }
    }
    
    private func loadCurrentSettings() {
        speechRate = Double(speechManager.preferredSpeechRate)
        
        // Set the voice index based on current preference
        if let preferredId = speechManager.preferredVoiceIdentifier,
           let index = availableVoices.firstIndex(where: { $0.identifier == preferredId }) {
            selectedVoiceIndex = index
        } else {
            setDefaultVoice()
        }
    }
    
    private func setDefaultVoice() {
        // First try to find Samantha voice
        if let samanthaIndex = availableVoices.firstIndex(where: { voice in
            voice.name.lowercased().contains("samantha")
        }) {
            selectedVoiceIndex = samanthaIndex
            return
        }
        
        // If Samantha not found, try to find a high-quality US English voice
        if let defaultUSIndex = availableVoices.firstIndex(where: { voice in
            voice.language == "en-US" && voice.quality == .enhanced
        }) {
            selectedVoiceIndex = defaultUSIndex
            return
        }
        
        // Fallback to first US English voice
        if let usIndex = availableVoices.firstIndex(where: { voice in
            voice.language == "en-US"
        }) {
            selectedVoiceIndex = usIndex
            return
        }
        
        // Final fallback to first available voice
        selectedVoiceIndex = 0
    }
    
    private func voiceDisplayName(for voice: AVSpeechSynthesisVoice) -> String {
        let languageCode = voice.language
        let displayName = voice.name.isEmpty ? "System Voice" : voice.name
        
        let qualityIndicator = voice.quality == .enhanced ? "(enhanced)" : ""
        
        return "\(displayName)\(qualityIndicator) (\(languageCode))"
    }
}

#Preview {
    SpeechSettingsView(speechManager: SpeechManager())
}
