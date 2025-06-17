import AVFoundation
import SwiftUI

@MainActor
class SpeechManager: ObservableObject {
    private let synthesizer = AVSpeechSynthesizer()
    @Published var isSpeaking: Bool = false
    @Published var isPaused: Bool = false
    
    @Published var autoReadResponses: Bool {
        didSet {
            UserDefaults.standard.set(autoReadResponses, forKey: "autoReadResponses")
        }
    }
    
    @Published var preferredVoiceIdentifier: String? {
        didSet {
            UserDefaults.standard.set(preferredVoiceIdentifier, forKey: "preferredVoiceIdentifier")
        }
    }
    
    @Published var preferredSpeechRate: Float {
        didSet {
            UserDefaults.standard.set(preferredSpeechRate, forKey: "preferredSpeechRate")
        }
    }
    
    private let speechDelegate: SpeechDelegate
    
    private var shouldStartAutoSpeech = false
    private var isGenerationComplete = false
    private var finalTextToSpeak: String = ""
    
    private var hasTriggeredAutoSpeechForCurrentResponse = false
    private var lastResponseStartTime: Date?
    
    init() {
        self.autoReadResponses = UserDefaults.standard.bool(forKey: "autoReadResponses")
        self.preferredVoiceIdentifier = UserDefaults.standard.string(forKey: "preferredVoiceIdentifier")
        self.preferredSpeechRate = UserDefaults.standard.object(forKey: "preferredSpeechRate") as? Float ?? 0.5
        
        self.speechDelegate = SpeechDelegate()
        setupAudioSession()
        synthesizer.delegate = speechDelegate
        speechDelegate.manager = self
    }
    
    deinit {
        synthesizer.stopSpeaking(at: .immediate)
        synthesizer.delegate = nil
    }
    
    private func setupAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
    
    func speak(_ text: String, rate: Float? = nil, voice: AVSpeechSynthesisVoice? = nil) {
        // Stop any current speech
        stop()
        
        // Clean the text for better speech
        let cleanText = cleanTextForSpeech(text)
        
        guard !cleanText.isEmpty else { return }
        
        let utterance = AVSpeechUtterance(string: cleanText)
        utterance.rate = rate ?? preferredSpeechRate // Use preferred rate if none specified
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        
        // Use preferred voice if available, otherwise use specified voice or system default
        if let voice = voice {
            utterance.voice = voice
        } else if let preferredId = preferredVoiceIdentifier,
                  let preferredVoice = AVSpeechSynthesisVoice(identifier: preferredId) {
            utterance.voice = preferredVoice
        } else {
            // Try to use Samantha or high-quality voice as default
            utterance.voice = getDefaultVoice()
        }
        
        isSpeaking = true
        isPaused = false
        
        synthesizer.speak(utterance)
    }
    
    func speakIfAutoEnabled(_ text: String) {
        if autoReadResponses {
            speak(text)
        }
    }
    
    func handleResponseUpdate(_ text: String, isFirstToken: Bool = false, isComplete: Bool = false) {
        if autoReadResponses {
            if isFirstToken {
                // Reset flags for new response
                shouldStartAutoSpeech = true
                isGenerationComplete = false
                finalTextToSpeak = ""
            }
            
            // Store the latest text
            finalTextToSpeak = text
            
            if isComplete {
                // Generation is complete, now speak the final text
                isGenerationComplete = true
                if shouldStartAutoSpeech && !isSpeaking {
                    speak(finalTextToSpeak)
                    shouldStartAutoSpeech = false // Prevent multiple triggers
                }
            } else if shouldStartAutoSpeech && !isSpeaking && shouldStartSpeaking(text) {
                // Only start speaking if we haven't started yet and content is meaningful
                // But don't speak during generation - wait for completion
                // We'll just mark that we should speak when complete
            }
        }
    }
    
    func handleGenerationComplete(_ finalText: String) {
        if autoReadResponses && shouldStartAutoSpeech && !isSpeaking {
            speak(finalText)
            shouldStartAutoSpeech = false
        }
    }
    
    func resetAutoSpeechFlag() {
        shouldStartAutoSpeech = false
        isGenerationComplete = false
        finalTextToSpeak = ""
    }
    
    private func shouldStartSpeaking(_ text: String) -> Bool {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Don't speak if text is too short (less than 10 characters)
        guard trimmedText.count >= 10 else { return false }
        
        // Don't speak if it's just a few words without a complete thought
        let words = trimmedText.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        guard words.count >= 3 else { return false }
        
        // Look for sentence endings or meaningful content indicators
        let hasCompleteSentence = trimmedText.contains(".") || trimmedText.contains("!") || trimmedText.contains("?")
        let hasSubstantialContent = trimmedText.count >= 20
        
        return hasCompleteSentence || hasSubstantialContent
    }
    
    func updatePreferences(voice: AVSpeechSynthesisVoice?, rate: Float) {
        preferredVoiceIdentifier = voice?.identifier
        preferredSpeechRate = rate
    }
    
    private func getDefaultVoice() -> AVSpeechSynthesisVoice? {
        let availableVoices = getAvailableVoices()
        
        // Try to find Samantha
        if let samantha = availableVoices.first(where: { $0.name.lowercased().contains("samantha") }) {
            return samantha
        }
        
        // Try enhanced US English voice
        if let enhanced = availableVoices.first(where: { $0.language == "en-US" && $0.quality == .enhanced }) {
            return enhanced
        }
        
        // Fallback to any US English voice
        if let usVoice = availableVoices.first(where: { $0.language == "en-US" }) {
            return usVoice
        }
        
        // System default
        return AVSpeechSynthesisVoice(language: "en-US") ?? AVSpeechSynthesisVoice(language: "en")
    }
    
    func pause() {
        guard isSpeaking && !isPaused else { return }
        synthesizer.pauseSpeaking(at: .immediate)
        isPaused = true
    }
    
    func resume() {
        guard isSpeaking && isPaused else { return }
        synthesizer.continueSpeaking()
        isPaused = false
    }
    
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
        isPaused = false
    }
    
    private func cleanTextForSpeech(_ text: String) -> String {
        var cleanText = text
        
        // Remove excessive whitespace and newlines
        cleanText = cleanText.trimmingCharacters(in: .whitespacesAndNewlines)
        cleanText = cleanText.replacingOccurrences(of: "\n+", with: ". ", options: .regularExpression)
        cleanText = cleanText.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        
        // Add pauses for better speech flow
        cleanText = cleanText.replacingOccurrences(of: ". ", with: ". ")
        cleanText = cleanText.replacingOccurrences(of: ", ", with: ", ")
        
        return cleanText
    }
    
    // Get available voices for the user to choose from
    func getAvailableVoices() -> [AVSpeechSynthesisVoice] {
        return AVSpeechSynthesisVoice.speechVoices().filter { voice in
            voice.language.hasPrefix("en") // Filter for English voices
        }.sorted { $0.name < $1.name }
    }
    
    // Convenience method to get voice names for UI
    func getVoiceNames() -> [String] {
        return getAvailableVoices().map { "\($0.name) (\($0.language))" }
    }
}

// Delegate to handle speech events with safer memory management
private class SpeechDelegate: NSObject, AVSpeechSynthesizerDelegate {
    weak var manager: SpeechManager?
    
    init(manager: SpeechManager? = nil) {
        self.manager = manager
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            manager?.isSpeaking = false
            manager?.isPaused = false
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            manager?.isSpeaking = false
            manager?.isPaused = false
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didPause utterance: AVSpeechUtterance) {
        Task { @MainActor in
            manager?.isPaused = true
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didContinue utterance: AVSpeechUtterance) {
        Task { @MainActor in
            manager?.isPaused = false
        }
    }
}
